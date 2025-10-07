from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psutil

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import orjson
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    orjson = None


@dataclass
class PDFMetric:
    pdf: str
    worker: str
    start_monotonic: float
    end_monotonic: float
    duration_seconds: float
    attempts: int
    success: bool
    start_wall_time: float
    end_wall_time: float
    error: Optional[str] = None


@dataclass
class GPUSample:
    timestamp: float
    utilization_percent: float
    memory_used_mb: float
    temperature_c: Optional[float]


@dataclass
class CPUSample:
    timestamp: float
    per_core_percent: List[float]
    system_memory_percent: float


class MetricsCollector:
    def __init__(
        self,
        *,
        output_dir: Path,
        sample_interval: float = 5.0,
        gpu_index: int = 0,
        benchmark_mode: bool = False,
    ):
        self.output_dir = output_dir
        self.sample_interval = sample_interval
        self.gpu_index = gpu_index
        self.benchmark_mode = benchmark_mode

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, name="metrics-sampler", daemon=True)
        self._lock = threading.Lock()

        self._active: Dict[Tuple[str, str], Tuple[float, int, float]] = {}
        self._records: List[PDFMetric] = []
        self._gpu_samples: List[GPUSample] = []
        self._cpu_samples: List[CPUSample] = []
        self._start_wall: Optional[float] = None

    # ------------------------------------------------------------------

    def start(self) -> None:
        self._start_wall = time.perf_counter()
        psutil.cpu_percent(interval=None, percpu=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------

    def record_pdf_start(
        self,
        pdf: str,
        worker: str,
        attempt: int,
        *,
        start_monotonic: Optional[float] = None,
        start_wall: Optional[float] = None,
    ) -> None:
        monotonic = start_monotonic if start_monotonic is not None else time.perf_counter()
        wall = start_wall if start_wall is not None else time.time()
        with self._lock:
            self._active[(pdf, worker)] = (monotonic, attempt, wall)

    def record_pdf_end(
        self,
        pdf: str,
        worker: str,
        success: bool,
        attempts: int,
        *,
        error: Optional[str] = None,
    ) -> None:
        key = (pdf, worker)
        with self._lock:
            start_info = self._active.pop(key, (time.perf_counter(), attempts, time.time()))
        start_monotonic = start_info[0]
        start_wall = start_info[2]
        end_monotonic = time.perf_counter()
        end_wall = time.time()
        duration = end_monotonic - start_monotonic

        with self._lock:
            self._records.append(
                PDFMetric(
                    pdf=pdf,
                    worker=worker,
                    start_monotonic=start_monotonic,
                    end_monotonic=end_monotonic,
                    duration_seconds=duration,
                    attempts=attempts,
                    success=success,
                    start_wall_time=start_wall,
                    end_wall_time=end_wall,
                    error=error,
                )
            )

    # ------------------------------------------------------------------

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self._collect_cpu_sample()
            self._collect_gpu_sample()
            self._stop_event.wait(self.sample_interval)

    def _collect_cpu_sample(self) -> None:
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        mem = psutil.virtual_memory()
        sample = CPUSample(
            timestamp=time.time(),
            per_core_percent=per_core,
            system_memory_percent=mem.percent,
        )
        with self._lock:
            self._cpu_samples.append(sample)

    def _collect_gpu_sample(self) -> None:
        nvidia_smi = shutil_which("nvidia-smi")
        if nvidia_smi is None:
            return

        query = [
            nvidia_smi,
            "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
            f"--id={self.gpu_index}",
        ]
        try:
            result = subprocess.run(
                query,
                capture_output=True,
                text=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return

        line = result.stdout.strip().splitlines()
        if not line:
            return
        fields = [item.strip() for item in line[0].split(",")]
        try:
            util = float(fields[0])
            mem = float(fields[1])
            temp = float(fields[2]) if len(fields) > 2 else None
        except (ValueError, IndexError):
            return

        sample = GPUSample(
            timestamp=time.time(),
            utilization_percent=util,
            memory_used_mb=mem,
            temperature_c=temp,
        )
        with self._lock:
            self._gpu_samples.append(sample)

        if temp is not None:
            LOGGER.debug("GPU temp: %.1f°C", temp)
            if temp >= 85:
                LOGGER.warning(
                    "GPU temperature %.1f°C exceeds threshold; consider switching to latency profile.",
                    temp,
                )

    # ------------------------------------------------------------------

    def generate_report(
        self,
        *,
        summary: "BatchSummary",
        profile: str,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "performance_report.json"
        with self._lock:
            pdf_data = [metric.__dict__ for metric in self._records]
            gpu_samples = [sample.__dict__ for sample in self._gpu_samples]
            cpu_samples = [sample.__dict__ for sample in self._cpu_samples]

        total_duration = max(summary.duration_seconds, 1e-9)
        throughput = (summary.succeeded / total_duration) * 3600 if total_duration else 0.0
        average_pdf_duration = (
            sum(metric.duration_seconds for metric in self._records if metric.success) / summary.succeeded
            if summary.succeeded
            else 0.0
        )
        avg_gpu_util = (
            sum(sample["utilization_percent"] for sample in gpu_samples) / len(gpu_samples)
            if gpu_samples
            else 0.0
        )
        max_gpu_mem = max((sample["memory_used_mb"] for sample in gpu_samples), default=0.0)

        report = {
            "profile": profile,
            "benchmark_mode": self.benchmark_mode,
            "summary": {
                "total": summary.total,
                "processed": summary.processed,
                "succeeded": summary.succeeded,
                "failed": summary.failed,
                "duration_seconds": summary.duration_seconds,
                "throughput_pdfs_per_hour": throughput,
                "avg_seconds_per_pdf": average_pdf_duration,
            },
            "pdf_metrics": pdf_data,
            "gpu_samples": gpu_samples,
            "cpu_samples": cpu_samples,
            "aggregates": {
                "average_gpu_utilization": avg_gpu_util,
                "max_gpu_memory_mb": max_gpu_mem,
            },
        }

        _write_json(report_path, report)

        LOGGER.info("Performance report written to %s", report_path)
        return report_path


def load_performance_report(path: Path) -> Dict[str, Any]:
    data = path.read_bytes()
    if orjson is not None:
        return orjson.loads(data)
    return json.loads(data.decode("utf-8"))


def compare_performance_reports(
    reports: Sequence[Path | str],
    *,
    output_path: Optional[Path] = None,
) -> Path:
    if len(reports) < 2:
        raise ValueError("At least two reports are required for comparison")

    normalized: List[Path] = [Path(report).expanduser().resolve() for report in reports]
    runs: List[Dict[str, Any]] = []
    best_throughput: Optional[Dict[str, Any]] = None
    best_latency: Optional[Dict[str, Any]] = None

    for report_path in normalized:
        data = load_performance_report(report_path)
        summary = data.get("summary", {})
        throughput = float(summary.get("throughput_pdfs_per_hour", 0.0))
        avg_seconds = float(summary.get("avg_seconds_per_pdf", 0.0))
        run_entry = {
            "path": str(report_path),
            "profile": data.get("profile"),
            "benchmark_mode": data.get("benchmark_mode", False),
            "throughput_pdfs_per_hour": throughput,
            "avg_seconds_per_pdf": avg_seconds,
            "succeeded": summary.get("succeeded"),
            "failed": summary.get("failed"),
            "total": summary.get("total"),
            "duration_seconds": summary.get("duration_seconds"),
        }
        runs.append(run_entry)

        if best_throughput is None or throughput > best_throughput["throughput_pdfs_per_hour"]:
            best_throughput = run_entry
        if (
            best_latency is None
            or (avg_seconds > 0 and avg_seconds < best_latency.get("avg_seconds_per_pdf", float("inf")))
        ):
            best_latency = run_entry

    comparison = {
        "runs": runs,
        "best_throughput": best_throughput,
        "best_latency": best_latency,
    }

    base_dir = output_path.parent if output_path else normalized[0].parent
    base_dir.mkdir(parents=True, exist_ok=True)
    destination = output_path or (base_dir / "performance_comparison.json")
    _write_json(destination, comparison)
    LOGGER.info("Performance comparison written to %s", destination)
    return destination


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    if orjson is not None:
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2) + b"\n")
    else:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


def shutil_which(command: str) -> Optional[str]:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / command
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


__all__ = [
    "MetricsCollector",
    "PDFMetric",
    "GPUSample",
    "CPUSample",
    "compare_performance_reports",
    "load_performance_report",
]
