from __future__ import annotations

import gc
import importlib
import importlib.util
import json
import logging
import multiprocessing as mp
import os
import queue
try:  # pragma: no cover - resource is Unix specific
    import resource
except ImportError:  # pragma: no cover
    resource = None
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timezone

import psutil

from .markdown_builder import MarkdownGenerationError, generate_structured_markdown
from .metrics import MetricsCollector
from .mineru_config import load_config, write_config
from .progress import ProgressBar
from .worker_coordinator import CoordinatorConfig, WorkerCoordinator, ensure_directories

LOGGER = logging.getLogger(__name__)


class BatchProcessorError(RuntimeError):
    """Raised when batch processing cannot continue."""


class WorkerShutdown(Exception):
    """Internal exception used to break worker loop."""


@dataclass(frozen=True)
class BatchSummary:
    total: int
    processed: int
    succeeded: int
    failed: int
    skipped: int
    duration_seconds: float
    failures: List[Dict[str, object]] = field(default_factory=list)
    failure_report: Optional[Path] = None


@dataclass
class BatchProcessorConfig:
    input_dir: Path
    output_dir: Path
    workers: int = max(1, (os.cpu_count() or 4) - 2)
    poll_interval: float = 2.0
    max_retries: int = 3
    retry_delay: float = 10.0
    progress_interval: float = 5.0
    mineru_cli: str = "mineru"
    mineru_backend: str = "vlm-vllm-engine"
    mineru_extra_args: Sequence[str] = field(default_factory=tuple)
    env_overrides: Dict[str, str] = field(default_factory=dict)
    log_progress: bool = True
    memory_pause_threshold: float = 0.80
    memory_resume_threshold: float = 0.70
    cpu_pause_threshold: float = 0.90
    resource_monitor_interval: float = 5.0
    profile: str = "balanced"
    max_workers: int = 14
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    max_model_len: int = 16384
    block_size: int = 16
    swap_space_mb: int = 8192
    dtype: str = "bfloat16"
    io_buffer_size: int = 10 * 1024 * 1024
    enable_cpu_affinity: bool = True
    reserved_system_cores: int = 2
    worker_memory_limit_gb: float = 12.0
    worker_niceness: int = 5
    omp_threads: int = 1
    mkl_threads: int = 1
    cpu_affinity_plan: Dict[int, List[int]] = field(default_factory=dict)
    benchmark: bool = False
    performance_report_path: Optional[Path] = None

    def __post_init__(self) -> None:
        input_dir = Path(self.input_dir).expanduser().resolve()
        output_dir = Path(self.output_dir).expanduser().resolve()
        object.__setattr__(self, "input_dir", input_dir)
        object.__setattr__(self, "output_dir", output_dir)
        if self.performance_report_path is not None:
            report_path = Path(self.performance_report_path).expanduser().resolve()
            object.__setattr__(self, "performance_report_path", report_path)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.workers > self.max_workers:
            raise ValueError("workers exceeds max_workers")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        if self.progress_interval <= 0:
            raise ValueError("progress_interval must be > 0")
        if not (0 < self.memory_resume_threshold < self.memory_pause_threshold < 1):
            raise ValueError("memory thresholds must satisfy 0 < resume < pause < 1")
        if not (0 < self.cpu_pause_threshold <= 1):
            raise ValueError("cpu_pause_threshold must be between 0 and 1")
        if self.resource_monitor_interval <= 0:
            raise ValueError("resource_monitor_interval must be > 0")
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        if self.io_buffer_size < 1_048_576:
            raise ValueError("io_buffer_size must be at least 1MB")
        if self.worker_memory_limit_gb <= 0:
            raise ValueError("worker_memory_limit_gb must be positive")
        if self.reserved_system_cores < 0:
            raise ValueError("reserved_system_cores must be >= 0")


def _load_optional_module(name: str):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    return importlib.import_module(name)


def _environment_snapshot(config: BatchProcessorConfig) -> Dict[str, str]:
    status: Dict[str, str] = {
        "backend": config.mineru_backend,
        "workers": str(config.workers),
    }
    torch_module = _load_optional_module("torch")
    if torch_module is None:
        status["cuda"] = "torch missing"
    elif torch_module.cuda.is_available() and torch_module.cuda.device_count() > 0:
        status["cuda"] = torch_module.cuda.get_device_name(0)
    else:
        status["cuda"] = "unavailable"

    status["vllm_module"] = "available" if importlib.util.find_spec("vllm") else "missing"
    cli_path = shutil.which(config.mineru_cli)
    status["cli"] = cli_path or "not found"
    return status


def _resolve_executable(executable: str) -> Optional[str]:
    """Return the resolved path for *executable* when it is runnable."""

    resolved = shutil.which(executable)
    if resolved:
        return resolved

    candidate = Path(executable)
    if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate.resolve())

    return None


def _relative_to(path: Path, parent: Path) -> Path:
    try:
        return path.relative_to(parent)
    except ValueError as exc:  # pragma: no cover - defensive
        raise BatchProcessorError(
            f"Path {path} is not inside {parent}. Check configuration."
        ) from exc


def _worker_env(config: BatchProcessorConfig) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(config.env_overrides)
    env.setdefault("OMP_NUM_THREADS", str(config.omp_threads))
    env.setdefault("MKL_NUM_THREADS", str(config.mkl_threads))
    env.setdefault("MINERU_VLLM_GPU_MEMORY_UTILIZATION", f"{config.gpu_memory_utilization:.2f}")
    env.setdefault("MINERU_VLLM_TENSOR_PARALLEL_SIZE", str(config.tensor_parallel_size))
    env.setdefault("MINERU_VLLM_DATA_PARALLEL_SIZE", str(config.data_parallel_size))
    env.setdefault("MINERU_VLLM_MAX_MODEL_LEN", str(config.max_model_len))
    env.setdefault("MINERU_VLLM_BLOCK_SIZE", str(config.block_size))
    env.setdefault("MINERU_VLLM_SWAP_SPACE_MB", str(config.swap_space_mb))
    env.setdefault("MINERU_VLLM_DTYPE", config.dtype)
    return env


def _mineru_command(
    *,
    cli: str,
    pdf_path: Path,
    output_dir: Path,
    backend: str,
    extra_args: Sequence[str],
) -> List[str]:
    command: List[str] = [
        cli,
        "-p",
        str(pdf_path),
        "-o",
        str(output_dir),
        "-b",
        backend,
    ]
    command.extend(extra_args)
    return command


def _buffered_copy(source_path: Path, destination: Path, *, buffer_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb", buffering=buffer_size) as src, destination.open(
        "wb", buffering=buffer_size
    ) as dst:
        shutil.copyfileobj(src, dst, length=buffer_size)
    shutil.copystat(source_path, destination, follow_symlinks=False)


def _move_outputs(
    *,
    temp_dir: Path,
    pdf_path: Path,
    config: BatchProcessorConfig,
) -> Tuple[List[Path], List[Path]]:
    """Move MinerU outputs from *temp_dir* into the final output directory.

    Returns tuple (written_files, artifact_files).
    """

    relative_pdf = _relative_to(pdf_path, config.input_dir)
    stem = relative_pdf.with_suffix("")
    base_dir = config.output_dir / stem.parent
    document_dir = base_dir
    artifacts_dir = base_dir / f"{stem.name}_artifacts"
    ensure_directories([document_dir, artifacts_dir])

    written: List[Path] = []
    artifacts: List[Path] = []
    written_set: set[Path] = set()

    def move_to(target_dir: Path, source_path: Path) -> Path:
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / source_path.name
        if destination.exists():
            if destination.is_file():
                destination.unlink()
            else:
                shutil.rmtree(destination)
        if source_path.is_file():
            _buffered_copy(source_path, destination, buffer_size=config.io_buffer_size)
            source_path.unlink(missing_ok=True)
        else:
            shutil.move(str(source_path), destination)
        return destination

    for source in temp_dir.iterdir():
        destination = move_to(artifacts_dir, source)
        artifacts.append(destination)

    document_dir.mkdir(parents=True, exist_ok=True)

    def first_match(pattern: str) -> Optional[Path]:
        matches = sorted(artifacts_dir.rglob(pattern))
        return matches[0] if matches else None

    markdown_source = first_match("*.md")
    markdown_dest: Optional[Path] = None
    if markdown_source:
        markdown_dest = document_dir / f"{stem.name}.md"
        shutil.copy2(markdown_source, markdown_dest)
        if markdown_dest not in written_set:
            written.append(markdown_dest)
            written_set.add(markdown_dest)

    structured_files = {
        "*_content_list.json": "content_list.json",
        "*_middle.json": "middle.json",
        "*_model.json": "model.json",
    }
    content_list_destination: Optional[Path] = None
    for pattern, relative_name in structured_files.items():
        source = first_match(pattern)
        if source is None:
            continue
        destination = document_dir / relative_name
        shutil.copy2(source, destination)
        doc_specific_destination = document_dir / f"{stem.name}_{relative_name}"
        if doc_specific_destination != destination:
            shutil.copy2(source, doc_specific_destination)
            if doc_specific_destination not in written_set:
                written.append(doc_specific_destination)
                written_set.add(doc_specific_destination)
        if pattern == "*_content_list.json":
            content_list_destination = (
                doc_specific_destination if doc_specific_destination.exists() else destination
            )
        if destination not in written_set:
            written.append(destination)
            written_set.add(destination)

    if content_list_destination is not None:
        structured_output = document_dir / f"{stem.name}.structured.md"
        try:
            structured_path = generate_structured_markdown(
                content_list_destination, output_path=structured_output
            )
        except FileNotFoundError as exc:
            LOGGER.error("Failed to load content list for %s: %s", stem.name, exc)
        except MarkdownGenerationError as exc:
            LOGGER.error(
                "Failed to generate structured Markdown for %s: %s", stem.name, exc
            )
        except Exception:  # pragma: no cover - unexpected
            LOGGER.exception(
                "Unexpected error during structured Markdown generation for %s",
                stem.name,
            )
        else:
            if structured_path not in written_set:
                written.append(structured_path)
                written_set.add(structured_path)

    artifact_files = {
        "*_layout.pdf": "layout.pdf",
        "*_origin.pdf": "origin.pdf",
    }
    for pattern, relative_name in artifact_files.items():
        source = first_match(pattern)
        if source is None:
            continue
        destination = document_dir / relative_name
        shutil.copy2(source, destination)
        artifacts.append(destination)
        doc_specific_destination = document_dir / f"{stem.name}_{relative_name}"
        if doc_specific_destination != destination:
            shutil.copy2(source, doc_specific_destination)
            artifacts.append(doc_specific_destination)

    if markdown_dest is None or not markdown_dest.exists():
        artifacts_snapshot: List[str] = []
        for index, candidate in enumerate(artifacts_dir.rglob("*")):
            artifacts_snapshot.append(str(candidate))
            if index >= 9:
                break
        raise BatchProcessorError(
            f"MinerU produced no Markdown output for {pdf_path.name}; "
            f"inspected {artifacts_snapshot[:10]}..."
        )

    return written, artifacts


def _process_pdf_once(
    *,
    config: BatchProcessorConfig,
    pdf_path: Path,
    worker_id: str,
    temp_root: Path,
) -> Tuple[List[Path], List[Path]]:
    temp_dir = temp_root / f"{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    command = _mineru_command(
        cli=config.mineru_cli,
        pdf_path=pdf_path,
        output_dir=temp_dir,
        backend=config.mineru_backend,
        extra_args=config.mineru_extra_args,
    )

    LOGGER.info("[%s] Running MinerU: %s", worker_id, " ".join(command))
    env = _worker_env(config)

    completed = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        log_excerpt = completed.stderr.strip() or completed.stdout.strip()
        message = log_excerpt.splitlines()[-1] if log_excerpt else "Unknown error"
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        ) from BatchProcessorError(message)

    written, artifacts = _move_outputs(temp_dir=temp_dir, pdf_path=pdf_path, config=config)

    shutil.rmtree(temp_dir, ignore_errors=True)
    return written, artifacts


def _cleanup_temp(temp_root: Path) -> None:
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)


def _worker_loop(
    worker_id: int,
    config_dict: Dict[str, object],
    status_queue: mp.Queue,
    shutdown_event: mp.Event,
    throttle_event: mp.Event,
    shared_stats,
) -> None:
    config = _config_from_dict(config_dict)
    coordinator = WorkerCoordinator(CoordinatorConfig(config.input_dir))
    temp_root = Path(tempfile.mkdtemp(prefix=f"mineru-worker-{worker_id}-"))
    ensure_directories([config.output_dir])

    if resource and config.worker_memory_limit_gb:
        limit_bytes = int(config.worker_memory_limit_gb * (1024**3))
        try:  # pragma: no cover - depends on platform permissions
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except (ValueError, OSError):
            LOGGER.debug("Worker %s unable to set memory limit", worker_id)

    if hasattr(os, "nice"):
        try:
            os.nice(config.worker_niceness)
        except OSError:
            LOGGER.debug("Worker %s unable to adjust niceness", worker_id)

    if config.enable_cpu_affinity and config.cpu_affinity_plan:
        cores = config.cpu_affinity_plan.get(worker_id)
        if cores:
            try:
                psutil.Process().cpu_affinity(cores)
            except (AttributeError, psutil.Error):
                LOGGER.debug("Worker %s unable to set CPU affinity", worker_id)

    try:
        idle_cycles = 0
        while not shutdown_event.is_set():
            while throttle_event.is_set() and not shutdown_event.is_set():
                time.sleep(config.poll_interval)

            pdf_path = coordinator.acquire_next(worker_id=str(worker_id))
            if pdf_path is None:
                idle_cycles += 1
                if idle_cycles % 5 == 0:
                    reclaimed = coordinator.clean_stale_locks()
                    if reclaimed:
                        status_queue.put({
                            "type": "info",
                            "worker": worker_id,
                            "message": f"Reclaimed {len(reclaimed)} stale locks",
                        })
                time.sleep(config.poll_interval)
                continue

            idle_cycles = 0
            attempt = 0
            success = False
            error_message = None
            pdf_start_wall = time.time()
            pdf_start_monotonic = time.perf_counter()
            status_queue.put({
                "type": "pdf_start",
                "worker": worker_id,
                "pdf": str(pdf_path),
                "start_wall": pdf_start_wall,
                "start_monotonic": pdf_start_monotonic,
                "attempt": attempt + 1,
            })
            while attempt <= config.max_retries and not success:
                try:
                    attempt += 1
                    written, artifacts = _process_pdf_once(
                        config=config,
                        pdf_path=pdf_path,
                        worker_id=f"worker-{worker_id}",
                        temp_root=temp_root,
                    )
                except subprocess.CalledProcessError as exc:
                    error_message = exc.stderr.strip() or str(exc)
                    status_queue.put({
                        "type": "retry",
                        "worker": worker_id,
                        "pdf": str(pdf_path),
                        "attempt": attempt,
                        "max_attempts": config.max_retries + 1,
                        "error": error_message,
                    })
                    if attempt > config.max_retries:
                        break
                    time.sleep(config.retry_delay)
                    continue
                except Exception as exc:  # pragma: no cover - safety
                    error_message = str(exc)
                    status_queue.put({
                        "type": "failure",
                        "worker": worker_id,
                        "pdf": str(pdf_path),
                        "error": error_message,
                    })
                    break
                else:
                    success = True
                    duration = time.perf_counter() - pdf_start_monotonic
                    with shared_stats.get_lock():
                        shared_stats[0] += 1
                        shared_stats[1] += 1
                    status_queue.put({
                        "type": "success",
                        "worker": worker_id,
                        "pdf": str(pdf_path),
                        "written": [str(p) for p in written],
                        "artifacts": [str(p) for p in artifacts],
                        "attempts": attempt,
                        "duration_seconds": duration,
                    })
                    status_queue.put({
                        "type": "pdf_finished",
                        "worker": worker_id,
                        "pdf": str(pdf_path),
                        "success": True,
                        "attempts": attempt,
                        "duration_seconds": duration,
                    })

            coordinator.release(
                pdf_path,
                worker_id=f"worker-{worker_id}",
                success=success,
                permanent=not success,
                message=error_message,
            )

            if not success:
                duration = time.perf_counter() - pdf_start_monotonic
                with shared_stats.get_lock():
                    shared_stats[0] += 1
                    shared_stats[2] += 1
                status_queue.put({
                    "type": "permanent_failure",
                    "worker": worker_id,
                    "pdf": str(pdf_path),
                    "error": error_message,
                    "attempts": attempt,
                    "duration_seconds": duration,
                })
                status_queue.put({
                    "type": "pdf_finished",
                    "worker": worker_id,
                    "pdf": str(pdf_path),
                    "success": False,
                    "attempts": attempt,
                    "duration_seconds": duration,
                    "error": error_message,
                })

            gc.collect()

        status_queue.put({"type": "worker_exit", "worker": worker_id})
    finally:
        _cleanup_temp(temp_root)


def _config_from_dict(config_dict: Dict[str, object]) -> BatchProcessorConfig:
    input_dir = Path(config_dict["input_dir"])  # type: ignore[arg-type]
    output_dir = Path(config_dict["output_dir"])  # type: ignore[arg-type]
    workers = int(config_dict["workers"])  # type: ignore[arg-type]
    poll_interval = float(config_dict["poll_interval"])  # type: ignore[arg-type]
    max_retries = int(config_dict["max_retries"])  # type: ignore[arg-type]
    retry_delay = float(config_dict["retry_delay"])  # type: ignore[arg-type]
    progress_interval = float(config_dict["progress_interval"])  # type: ignore[arg-type]
    mineru_cli = str(config_dict.get("mineru_cli", "mineru"))
    mineru_backend = str(config_dict["mineru_backend"])  # type: ignore[arg-type]
    mineru_extra_args = tuple(config_dict["mineru_extra_args"])  # type: ignore[arg-type]
    env_overrides = dict(config_dict["env_overrides"])  # type: ignore[arg-type]
    log_progress = bool(config_dict["log_progress"])  # type: ignore[arg-type]
    memory_pause_threshold = float(config_dict.get("memory_pause_threshold", 0.80))
    memory_resume_threshold = float(config_dict.get("memory_resume_threshold", 0.70))
    cpu_pause_threshold = float(config_dict.get("cpu_pause_threshold", 0.90))
    resource_monitor_interval = float(config_dict.get("resource_monitor_interval", 5.0))
    profile = str(config_dict.get("profile", "balanced"))
    max_workers = int(config_dict.get("max_workers", workers))
    gpu_memory_utilization = float(config_dict.get("gpu_memory_utilization", 0.90))
    tensor_parallel_size = int(config_dict.get("tensor_parallel_size", 1))
    data_parallel_size = int(config_dict.get("data_parallel_size", 1))
    max_model_len = int(config_dict.get("max_model_len", 16384))
    block_size = int(config_dict.get("block_size", 16))
    swap_space_mb = int(config_dict.get("swap_space_mb", 8192))
    dtype = str(config_dict.get("dtype", "bfloat16"))
    io_buffer_size = int(config_dict.get("io_buffer_size", 10 * 1024 * 1024))
    enable_cpu_affinity = bool(config_dict.get("enable_cpu_affinity", True))
    reserved_system_cores = int(config_dict.get("reserved_system_cores", 2))
    worker_memory_limit_gb = float(config_dict.get("worker_memory_limit_gb", 12.0))
    worker_niceness = int(config_dict.get("worker_niceness", 5))
    omp_threads = int(config_dict.get("omp_threads", 1))
    mkl_threads = int(config_dict.get("mkl_threads", 1))
    cpu_affinity_plan_raw = dict(config_dict.get("cpu_affinity_plan", {}))
    cpu_affinity_plan = {int(k): list(v) for k, v in cpu_affinity_plan_raw.items()}
    benchmark = bool(config_dict.get("benchmark", False))
    performance_report_path_raw = config_dict.get("performance_report_path")
    performance_report_path = (
        Path(performance_report_path_raw) if performance_report_path_raw else None
    )

    return BatchProcessorConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        workers=workers,
        poll_interval=poll_interval,
        max_retries=max_retries,
        retry_delay=retry_delay,
        progress_interval=progress_interval,
        mineru_cli=mineru_cli,
        mineru_backend=mineru_backend,
        mineru_extra_args=mineru_extra_args,
        env_overrides=env_overrides,
        log_progress=log_progress,
        memory_pause_threshold=memory_pause_threshold,
        memory_resume_threshold=memory_resume_threshold,
        cpu_pause_threshold=cpu_pause_threshold,
        resource_monitor_interval=resource_monitor_interval,
        profile=profile,
        max_workers=max_workers,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        max_model_len=max_model_len,
        block_size=block_size,
        swap_space_mb=swap_space_mb,
        dtype=dtype,
        io_buffer_size=io_buffer_size,
        enable_cpu_affinity=enable_cpu_affinity,
        reserved_system_cores=reserved_system_cores,
        worker_memory_limit_gb=worker_memory_limit_gb,
        worker_niceness=worker_niceness,
        omp_threads=omp_threads,
        mkl_threads=mkl_threads,
        cpu_affinity_plan=cpu_affinity_plan,
        benchmark=benchmark,
        performance_report_path=performance_report_path,
    )


class BatchProcessor:
    def __init__(self, config: BatchProcessorConfig):
        self.config = config
        self._mp_context = mp.get_context("spawn")
        self._start_time: Optional[float] = None
        self._shutdown_event = self._mp_context.Event()
        self._throttle_event = self._mp_context.Event()
        self._status_queue: mp.Queue = self._mp_context.Queue()
        self._workers: List[mp.Process] = []
        self._resource_thread: Optional[threading.Thread] = None
        self._metrics: Optional[MetricsCollector] = None
        self._shared_stats = self._mp_context.Array("i", [0, 0, 0])  # processed, success, failure

    # ------------------------------------------------------------------

    def run(self) -> BatchSummary:
        coordinator = WorkerCoordinator(CoordinatorConfig(self.config.input_dir))
        pending = coordinator.pending_items()
        total_jobs = len(pending)
        ensure_directories([self.config.output_dir])

        if total_jobs == 0:
            LOGGER.warning("No PDFs found in %s", self.config.input_dir)
            return BatchSummary(total=0, processed=0, succeeded=0, failed=0, skipped=0, duration_seconds=0.0)

        LOGGER.info(
            "Starting batch processing: %d PDFs, %d workers (profile=%s)",
            total_jobs,
            self.config.workers,
            self.config.profile,
        )

        metrics: Optional[MetricsCollector] = None
        metrics_dir: Optional[Path] = None
        if total_jobs > 0:
            metrics_dir = (
                self.config.performance_report_path.parent
                if self.config.performance_report_path
                else self.config.output_dir / "performance_metrics"
            )
            metrics = MetricsCollector(
                output_dir=metrics_dir,
                sample_interval=5.0,
                gpu_index=0,
                benchmark_mode=self.config.benchmark,
            )
            metrics.start()
            self._metrics = metrics

        environment_status = _environment_snapshot(self.config)
        progress_enabled = self.config.log_progress and total_jobs > 0

        with ProgressBar(
            total=6,
            desc="Setup",
            unit="step",
            enabled=progress_enabled,
            leave=True,
        ) as setup_bar:
            setup_bar.set_postfix(environment_status)
            setup_bar.update()

            preflight_status = self._run_preflight_checks()
            environment_status = {**environment_status, **preflight_status}
            setup_bar.set_postfix(environment_status)
            setup_bar.update()

            self._validate_resources()
            setup_bar.set_postfix({**environment_status, "resources": "ok"})
            setup_bar.update()

            self._configure_mineru()
            environment_status = {**environment_status, "dtype": self.config.dtype}
            setup_bar.set_postfix(environment_status)
            setup_bar.update()

            self._start_time = time.perf_counter()
            self._install_signal_handlers()
            setup_bar.set_postfix({**environment_status, "signals": "ready"})
            setup_bar.update()

            self._start_workers()
            self._start_resource_monitor()
            setup_bar.set_postfix({**environment_status, "workers": str(self.config.workers)})
            setup_bar.update()

        processed = succeeded = failed = skipped = 0
        last_progress_time = time.perf_counter()
        counted_documents: set[str] = set()
        worker_bars: Dict[str, ProgressBar] = {}
        failure_attempts: Dict[str, List[Dict[str, object]]] = {}
        failure_attempt_counts: Dict[str, int] = {}
        failure_errors: Dict[str, Optional[str]] = {}
        failure_order: List[str] = []

        def worker_position(label: str) -> int:
            try:
                return int(label) + 1
            except (ValueError, TypeError):
                return (len(worker_bars) % max(self.config.workers, 1)) + 1

        def record_attempt(label: str, attempt_number: int, error: Optional[str]) -> None:
            attempts = failure_attempts.setdefault(label, [])
            if attempt_number <= 0:
                attempt_number = len(attempts) + 1
            if attempts and attempts[-1].get("number") == attempt_number:
                if error is not None:
                    attempts[-1]["error"] = error
                return
            entry: Dict[str, object] = {"number": attempt_number}
            if error is not None:
                entry["error"] = error
            attempts.append(entry)

        def relative_label(label: str) -> str:
            pdf_path = Path(label)
            try:
                return str(pdf_path.relative_to(self.config.input_dir))
            except ValueError:
                try:
                    resolved = pdf_path.resolve()
                    return str(resolved.relative_to(self.config.input_dir))
                except Exception:  # pragma: no cover - fallback path handling
                    return pdf_path.name or str(pdf_path)

        try:
            with ProgressBar(
                total=total_jobs,
                desc="Documents",
                unit="doc",
                enabled=progress_enabled,
                leave=True,
                position=0,
            ) as overall_bar:
                overall_bar.set_postfix(
                    {
                        "success": succeeded,
                        "failed": failed,
                        "remaining": total_jobs,
                        "cuda": environment_status.get("cuda", "n/a"),
                        "vllm": environment_status.get("backend", self.config.mineru_backend),
                        "vllm_mod": environment_status.get("vllm_module", "unknown"),
                        "cli": environment_status.get("cli", "not found"),
                    }
                )

                while self._any_worker_alive():
                    try:
                        message = self._status_queue.get(timeout=self.config.progress_interval)
                    except queue.Empty:
                        message = None

                    if message:
                        msg_type = message.get("type")
                        worker_label = str(message.get("worker", ""))
                        pdf_label = str(message.get("pdf", ""))

                        if msg_type == "pdf_start":
                            if metrics:
                                metrics.record_pdf_start(
                                    pdf=pdf_label,
                                    worker=worker_label,
                                    attempt=int(message.get("attempt", 1)),
                                    start_monotonic=message.get("start_monotonic"),
                                    start_wall=message.get("start_wall"),
                                )
                            if progress_enabled:
                                bar = worker_bars.pop(worker_label, None)
                                if bar:
                                    bar.close()
                                description = f"{Path(pdf_label).name or 'document'} [{worker_label}]"
                                worker_bar = ProgressBar(
                                    total=1,
                                    desc=description,
                                    unit="doc",
                                    enabled=True,
                                    leave=False,
                                    position=worker_position(worker_label),
                                )
                                worker_bar.set_postfix(
                                    {
                                        "status": "processing",
                                        "attempt": int(message.get("attempt", 1)),
                                    }
                                )
                                worker_bars[worker_label] = worker_bar
                            continue

                        if msg_type == "pdf_finished":
                            if metrics:
                                metrics.record_pdf_end(
                                    pdf=pdf_label,
                                    worker=worker_label,
                                    success=bool(message.get("success", False)),
                                    attempts=int(message.get("attempts", 1)),
                                    error=message.get("error"),
                                )
                            if progress_enabled:
                                bar = worker_bars.pop(worker_label, None)
                                if bar:
                                    status_text = "success" if message.get("success") else "failed"
                                    postfix: Dict[str, object] = {
                                        "status": status_text,
                                        "attempts": int(message.get("attempts", 1)),
                                    }
                                    duration = message.get("duration_seconds")
                                    if duration is not None:
                                        postfix["secs"] = f"{float(duration):.2f}"
                                    if message.get("error"):
                                        postfix["error"] = str(message.get("error"))
                                    bar.set_postfix(postfix)  # type: ignore[arg-type]
                                    bar.update()
                                    bar.close()
                            continue

                        if msg_type == "success":
                            if pdf_label not in counted_documents:
                                counted_documents.add(pdf_label)
                                succeeded += 1
                                processed += 1
                                LOGGER.debug(
                                    "PDF processed: %s (worker %s)",
                                    pdf_label,
                                    worker_label,
                                )
                                overall_bar.update(1)
                                remaining = max(total_jobs - processed, 0)
                                overall_bar.set_postfix(
                                    {
                                        "success": succeeded,
                                        "failed": failed,
                                        "remaining": remaining,
                                        "cuda": environment_status.get("cuda", "n/a"),
                                        "vllm": environment_status.get(
                                            "backend", self.config.mineru_backend
                                        ),
                                        "vllm_mod": environment_status.get(
                                            "vllm_module", "unknown"
                                        ),
                                        "cli": environment_status.get("cli", "not found"),
                                    }
                                )
                            continue

                        if msg_type == "permanent_failure":
                            error_value = message.get("error")
                            error_text = (
                                str(error_value) if error_value is not None else None
                            )
                            attempts_value = int(message.get("attempts", 0))
                            record_attempt(pdf_label, attempts_value, error_text)
                            failure_attempt_counts[pdf_label] = (
                                attempts_value
                                if attempts_value > 0
                                else len(failure_attempts.get(pdf_label, []))
                            )
                            failure_errors[pdf_label] = error_text
                            if pdf_label not in failure_order:
                                failure_order.append(pdf_label)
                            if pdf_label not in counted_documents:
                                counted_documents.add(pdf_label)
                                failed += 1
                                processed += 1
                                LOGGER.error(
                                    "PDF failed: %s (worker %s) error=%s",
                                    pdf_label,
                                    worker_label,
                                    message.get("error"),
                                )
                                overall_bar.update(1)
                                remaining = max(total_jobs - processed, 0)
                                overall_bar.set_postfix(
                                    {
                                        "success": succeeded,
                                        "failed": failed,
                                        "remaining": remaining,
                                        "cuda": environment_status.get("cuda", "n/a"),
                                        "vllm": environment_status.get(
                                            "backend", self.config.mineru_backend
                                        ),
                                        "vllm_mod": environment_status.get(
                                            "vllm_module", "unknown"
                                        ),
                                        "cli": environment_status.get("cli", "not found"),
                                    }
                                )
                            continue

                        if msg_type == "failure":
                            error_value = message.get("error")
                            error_text = (
                                str(error_value) if error_value is not None else None
                            )
                            record_attempt(pdf_label, int(message.get("attempt", 0)), error_text)
                            LOGGER.error(
                                "PDF failed: %s (worker %s) error=%s",
                                pdf_label,
                                worker_label,
                                message.get("error"),
                            )
                            continue

                        if msg_type == "retry":
                            error_value = message.get("error")
                            error_text = (
                                str(error_value) if error_value is not None else None
                            )
                            record_attempt(pdf_label, int(message.get("attempt", 0)), error_text)
                            retry_message = (
                                f"Retry {message.get('attempt')}/{message.get('max_attempts')}"
                                f" for {pdf_label} due to {message.get('error')}"
                            )
                            overall_bar.write(retry_message)
                            LOGGER.warning(
                                "Retry %d/%d for %s due to %s",
                                message.get("attempt"),
                                message.get("max_attempts"),
                                pdf_label,
                                message.get("error"),
                            )
                            continue

                        if msg_type == "info":
                            info_message = f"Worker {worker_label}: {message.get('message')}"
                            overall_bar.write(info_message)
                            LOGGER.info("Worker %s: %s", worker_label, message.get("message"))
                            continue

                    now = time.perf_counter()
                    if (
                        self.config.log_progress
                        and now - last_progress_time >= self.config.progress_interval
                    ):
                        last_progress_time = now
                        remaining = max(total_jobs - processed, 0)
                        LOGGER.info(
                            "Progress: %d/%d processed (success=%d, failed=%d, remaining=%d)",
                            processed,
                            total_jobs,
                            succeeded,
                            failed,
                            remaining,
                        )

                    if processed >= total_jobs:
                        LOGGER.info("All jobs processed; signalling workers to stop")
                        self._shutdown_event.set()
                        break
        finally:
            for bar in worker_bars.values():
                bar.close()
            self._shutdown_event.set()
            self._join_workers()
            self._stop_resource_monitor()
            self._restore_signal_handlers()
            if metrics:
                metrics.stop()

        duration = time.perf_counter() - (self._start_time or time.perf_counter())

        failure_details: List[Dict[str, object]] = []
        for label in failure_order:
            attempts = [dict(item) for item in failure_attempts.get(label, [])]
            attempt_count = failure_attempt_counts.get(label, len(attempts))
            if attempt_count <= 0:
                attempt_count = len(attempts)
            final_error = failure_errors.get(label)
            entry: Dict[str, object] = {
                "pdf": relative_label(label),
                "attempt_count": attempt_count,
                "attempts": attempts,
                "final_error": final_error,
            }
            failure_details.append(entry)

        failure_report_path: Optional[Path] = None
        if failure_details:
            report_payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_failures": len(failure_details),
                "failures": failure_details,
            }
            failure_report_path = self.config.output_dir / "failed_documents.json"
            failure_report_path.parent.mkdir(parents=True, exist_ok=True)
            failure_report_path.write_text(
                json.dumps(report_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            LOGGER.info(
                "Failure report saved to %s (%d documents)",
                failure_report_path,
                len(failure_details),
            )
        else:
            LOGGER.info("No permanent failures encountered; skipping failure report generation")

        if failure_report_path:
            LOGGER.info(
                "Batch processing complete: %d succeeded, %d failed in %.2fs (failure report: %s)",
                succeeded,
                failed,
                duration,
                failure_report_path,
            )
        else:
            LOGGER.info(
                "Batch processing complete: %d succeeded, %d failed in %.2fs (no failures)",
                succeeded,
                failed,
                duration,
            )

        summary = BatchSummary(
            total=total_jobs,
            processed=processed,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            duration_seconds=duration,
            failures=failure_details,
            failure_report=failure_report_path,
        )

        if metrics and summary.processed > 0:
            report_path = metrics.generate_report(summary=summary, profile=self.config.profile)
            if (
                self.config.performance_report_path
                and report_path != self.config.performance_report_path
            ):
                self.config.performance_report_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(report_path, self.config.performance_report_path)
                report_path = self.config.performance_report_path
            LOGGER.info("Performance report saved to %s", report_path)

        stats_snapshot = list(self._shared_stats[:])
        LOGGER.debug(
            "Worker shared stats processed=%d success=%d failure=%d",
            *stats_snapshot,
        )

        return summary

    # ------------------------------------------------------------------

    def _start_workers(self) -> None:
        config_dict: Dict[str, object] = {
            "input_dir": str(self.config.input_dir),
            "output_dir": str(self.config.output_dir),
            "workers": self.config.workers,
            "poll_interval": self.config.poll_interval,
            "max_retries": self.config.max_retries,
            "retry_delay": self.config.retry_delay,
            "progress_interval": self.config.progress_interval,
            "mineru_cli": self.config.mineru_cli,
            "mineru_backend": self.config.mineru_backend,
            "mineru_extra_args": tuple(self.config.mineru_extra_args),
            "env_overrides": dict(self.config.env_overrides),
            "log_progress": self.config.log_progress,
            "memory_pause_threshold": self.config.memory_pause_threshold,
            "memory_resume_threshold": self.config.memory_resume_threshold,
            "cpu_pause_threshold": self.config.cpu_pause_threshold,
            "resource_monitor_interval": self.config.resource_monitor_interval,
            "profile": self.config.profile,
            "max_workers": self.config.max_workers,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "data_parallel_size": self.config.data_parallel_size,
            "max_model_len": self.config.max_model_len,
            "block_size": self.config.block_size,
            "swap_space_mb": self.config.swap_space_mb,
            "dtype": self.config.dtype,
            "io_buffer_size": self.config.io_buffer_size,
            "enable_cpu_affinity": self.config.enable_cpu_affinity,
            "reserved_system_cores": self.config.reserved_system_cores,
            "worker_memory_limit_gb": self.config.worker_memory_limit_gb,
            "worker_niceness": self.config.worker_niceness,
            "omp_threads": self.config.omp_threads,
            "mkl_threads": self.config.mkl_threads,
            "cpu_affinity_plan": dict(self.config.cpu_affinity_plan),
            "benchmark": self.config.benchmark,
            "performance_report_path": str(self.config.performance_report_path)
            if self.config.performance_report_path
            else None,
        }

        for worker_id in range(self.config.workers):
            process = self._mp_context.Process(
                target=_worker_loop,
                args=(
                    worker_id,
                    config_dict,
                    self._status_queue,
                    self._shutdown_event,
                    self._throttle_event,
                    self._shared_stats,
                ),
                daemon=True,
            )
            process.start()
            self._workers.append(process)
            LOGGER.debug("Started worker process %s (pid=%s)", worker_id, process.pid)
            self._apply_worker_policies(pid=process.pid, worker_id=worker_id)

    def _any_worker_alive(self) -> bool:
        return any(worker.is_alive() for worker in self._workers)

    def _join_workers(self) -> None:
        for worker in self._workers:
            worker.join(timeout=5)
            if worker.is_alive():
                LOGGER.warning("Terminating unresponsive worker pid=%s", worker.pid)
                worker.terminate()
        self._workers.clear()

    # ------------------------------------------------------------------

    def _apply_worker_policies(self, *, pid: int, worker_id: int) -> None:
        try:
            proc = psutil.Process(pid)
        except psutil.Error:
            return

        if self.config.enable_cpu_affinity:
            cores = self.config.cpu_affinity_plan.get(worker_id)
            if not cores and (os.cpu_count() or 0) > self.config.reserved_system_cores:
                total_cores = max((os.cpu_count() or 0) - self.config.reserved_system_cores, 1)
                base_core = self.config.reserved_system_cores + (worker_id % total_cores)
                cores = [base_core]
            if cores:
                try:
                    proc.cpu_affinity(cores)
                except (AttributeError, psutil.Error):
                    LOGGER.debug("Unable to set CPU affinity for worker %s", worker_id)

        try:
            proc.nice(self.config.worker_niceness)
        except (psutil.Error, AttributeError):
            LOGGER.debug("Unable to adjust niceness for worker %s", worker_id)

    def _configure_mineru(self) -> None:
        config = load_config()
        dtype = self._resolve_dtype_preference()
        try:
            config.update_vllm_settings(
                data_parallel_size=self.config.data_parallel_size,
                tensor_parallel_size=self.config.tensor_parallel_size,
                max_model_len=self.config.max_model_len,
                dtype=dtype,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                block_size=self.config.block_size,
                swap_space_mb=self.config.swap_space_mb,
            )
        except AttributeError:
            # For compatibility with existing configs lacking new fields
            config.vllm_settings.data_parallel_size = self.config.data_parallel_size
            config.vllm_settings.tensor_parallel_size = self.config.tensor_parallel_size
            config.vllm_settings.max_model_len = self.config.max_model_len
            config.vllm_settings.dtype = dtype
            config.vllm_settings.gpu_memory_utilization = self.config.gpu_memory_utilization
            config.vllm_settings.block_size = self.config.block_size
            config.vllm_settings.swap_space_mb = self.config.swap_space_mb

        write_config(config)
        self.config.dtype = dtype
        LOGGER.info(
            "Configured MinerU vLLM settings: gpu_mem_util=%.2f dtype=%s max_seq=%d",
            self.config.gpu_memory_utilization,
            dtype,
            self.config.max_model_len,
        )

    def _resolve_dtype_preference(self) -> str:
        preferred = self.config.dtype.lower()
        if preferred != "bfloat16":
            return preferred

        try:
            import torch

            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return "bfloat16"
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 8:
                return "bfloat16"
        except ModuleNotFoundError:
            LOGGER.debug("PyTorch not available; falling back to float16")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed dtype capability check: %s", exc)

        LOGGER.info("Falling back to float16 due to missing bfloat16 support")
        return "float16"

    def _run_preflight_checks(self) -> Dict[str, str]:
        """Validate environment dependencies before spawning workers."""

        status: Dict[str, str] = {}

        cli_path = _resolve_executable(self.config.mineru_cli)
        if not cli_path:
            raise BatchProcessorError(
                "MinerU CLI executable not found. Ensure 'mineru' is installed and "
                f"accessible as '{self.config.mineru_cli}'."
            )
        status["cli"] = cli_path

        missing_modules: List[str] = []
        for module_name in ("torch", "vllm"):
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError:
                missing_modules.append(module_name)
            except Exception as exc:  # pragma: no cover - defensive
                raise BatchProcessorError(
                    f"Failed to import required module '{module_name}': {exc}"
                ) from exc

        if missing_modules:
            module_list = ", ".join(sorted(missing_modules))
            raise BatchProcessorError(
                "Missing required Python modules: "
                f"{module_list}. Install them before running batch processing."
            )

        status["dependencies"] = "ok"

        try:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise BatchProcessorError(
                f"Failed to prepare output directory {self.config.output_dir}: {exc}"
            ) from exc

        probe_path = self.config.output_dir / f".mineru_write_test_{uuid.uuid4().hex}"
        try:
            probe_path.write_bytes(b"")
            probe_path.unlink()
        except Exception as exc:
            raise BatchProcessorError(
                f"Output directory {self.config.output_dir} is not writable: {exc}"
            ) from exc

        status["output"] = "ok"
        LOGGER.info(
            "Preflight validation succeeded: cli=%s dependencies=%s output_dir=%s",
            status["cli"],
            status["dependencies"],
            self.config.output_dir,
        )
        status.setdefault("preflight", "ok")
        return status

    def _validate_resources(self) -> None:
        total_ram = psutil.virtual_memory().total
        requested_ram = self.config.workers * self.config.worker_memory_limit_gb * 1024**3
        if requested_ram > total_ram * 0.90:
            raise BatchProcessorError(
                "Requested worker memory exceeds available system RAM. "
                "Reduce workers or memory per worker."
            )

        if self.config.workers > self.config.max_workers:
            raise BatchProcessorError(
                f"Requested workers ({self.config.workers}) exceed maximum {self.config.max_workers}."
            )

        try:
            import torch

            if torch.cuda.is_available():
                total_gpu = float(torch.cuda.get_device_properties(0).total_memory)
                reserved_gpu = total_gpu * self.config.gpu_memory_utilization
                if reserved_gpu > total_gpu * 0.98:
                    raise BatchProcessorError(
                        "GPU memory utilization setting leaves insufficient headroom."
                    )
        except ModuleNotFoundError:
            LOGGER.warning("PyTorch not available; skipping GPU resource validation")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("GPU validation skipped: %s", exc)

    def _install_signal_handlers(self) -> None:
        self._original_handlers: Dict[int, object] = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal_handler)

    def _restore_signal_handlers(self) -> None:
        if not hasattr(self, "_original_handlers"):
            return
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)

    def _signal_handler(self, sig: int, frame) -> None:  # pragma: no cover - requires signal
        LOGGER.warning("Received signal %s; requesting graceful shutdown", sig)
        self._shutdown_event.set()

    # ------------------------------------------------------------------

    def _start_resource_monitor(self) -> None:
        if self._resource_thread and self._resource_thread.is_alive():
            return
        self._resource_thread = threading.Thread(
            target=self._resource_monitor_loop,
            name="resource-monitor",
            daemon=True,
        )
        self._resource_thread.start()

    def _stop_resource_monitor(self) -> None:
        if self._resource_thread is None:
            return
        self._resource_thread.join(timeout=2)
        self._resource_thread = None

    def _resource_monitor_loop(self) -> None:
        while not self._shutdown_event.is_set():
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)

            if (
                mem.percent / 100 >= self.config.memory_pause_threshold
                or cpu / 100 >= self.config.cpu_pause_threshold
            ):
                if not self._throttle_event.is_set():
                    LOGGER.warning(
                        "Resource saturation detected (mem=%.1f%% cpu=%.1f%%); throttling",
                        mem.percent,
                        cpu,
                    )
                self._throttle_event.set()
            elif (
                self._throttle_event.is_set()
                and mem.percent / 100 <= self.config.memory_resume_threshold
                and cpu / 100 <= self.config.cpu_pause_threshold * 0.85
            ):
                LOGGER.info(
                    "Resources recovered (mem=%.1f%% cpu=%.1f%%); resuming",
                    mem.percent,
                    cpu,
                )
                self._throttle_event.clear()

            time.sleep(self.config.resource_monitor_interval)


__all__ = [
    "BatchProcessor",
    "BatchProcessorConfig",
    "BatchSummary",
    "discover_pdfs",
]
