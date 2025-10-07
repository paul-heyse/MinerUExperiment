from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .gpu_utils import GPUInfo, GPUUnavailableError, enforce_gpu_environment, warmup_gpu
from .mineru_config import MineruConfig, ModelDownloadError, ensure_model_downloaded, load_config
from .mineru_runner import (
    MineruInvocationError,
    MineruProcessResult,
    EXPECTED_OUTPUT_FILES,
    process_pdf,
)
from .progress import ProgressBar

LOGGER = logging.getLogger(__name__)


class ValidationFailure(RuntimeError):
    """Raised when validation cannot complete successfully."""


def _parse_cuda_devices(value: str) -> List[int]:
    devices: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            devices.append(int(part))
        except ValueError as exc:
            raise ValidationFailure(f"Invalid CUDA device index: {part}") from exc
    if not devices:
        devices.append(0)
    return devices


def _collect_gpu_snapshot() -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        LOGGER.warning("nvidia-smi not found; skipping GPU utilization snapshot.")
        return None

    if result.returncode != 0:
        LOGGER.warning("nvidia-smi returned %d: %s", result.returncode, result.stderr.strip())
        return None

    return result.stdout.strip()


@dataclass
class RunMetrics:
    duration_seconds: float
    result: MineruProcessResult

    def to_dict(self) -> Dict[str, object]:
        return {
            "duration_seconds": self.duration_seconds,
            "output_dir": str(self.result.output_dir),
            "output_files": {name: str(path) for name, path in self.result.output_files.items()},
        }


@dataclass
class ValidationReport:
    pdf_path: Path
    baseline: RunMetrics
    warmed: RunMetrics
    warmup_duration_seconds: float
    gpu_info: GPUInfo
    gpu_snapshot_before: Optional[str]
    gpu_snapshot_after: Optional[str]

    def warmed_is_faster(self) -> bool:
        return self.warmed.duration_seconds < self.baseline.duration_seconds

    def speedup_ratio(self) -> Optional[float]:
        if self.warmed.duration_seconds == 0:
            return None
        return self.baseline.duration_seconds / self.warmed.duration_seconds

    def to_dict(self) -> Dict[str, object]:
        return {
            "pdf_path": str(self.pdf_path),
            "baseline": self.baseline.to_dict(),
            "warmed": self.warmed.to_dict(),
            "warmup_duration_seconds": self.warmup_duration_seconds,
            "gpu_info": self.gpu_info.to_dict(),
            "gpu_snapshot_before": self.gpu_snapshot_before,
            "gpu_snapshot_after": self.gpu_snapshot_after,
            "warmed_is_faster": self.warmed_is_faster(),
            "speedup_ratio": self.speedup_ratio(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _ensure_success(result: MineruProcessResult) -> None:
    if result.success:
        return
    raise ValidationFailure(f"MinerU processing failed: {result.error or 'unknown error'}")


def _measure_processing_run(
    pdf_path: Path,
    *,
    output_dir: Path,
    config: MineruConfig,
    warmup: bool,
    expected_files: Sequence[str],
    show_progress: bool,
) -> RunMetrics:
    start = time.perf_counter()
    try:
        result = process_pdf(
            pdf_path,
            output_dir=output_dir,
            config=config,
            warmup=warmup,
            expected_files=expected_files,
            show_progress=show_progress,
        )
    except MineruInvocationError as exc:
        raise ValidationFailure(str(exc)) from exc
    duration = time.perf_counter() - start
    _ensure_success(result)
    return RunMetrics(duration_seconds=duration, result=result)


def validate_pdf_processing(
    pdf_path: Path | str,
    *,
    output_root: Optional[Path | str] = None,
    expected_files: Sequence[str] = EXPECTED_OUTPUT_FILES,
    show_progress: bool = True,
) -> ValidationReport:
    """
    Execute MinerU against a PDF twice (cold vs warmed) and capture metrics.

    This performs the following steps:
      * Ensures the MinerU model is downloaded.
      * Enforces GPU environment according to the configuration.
      * Records GPU info and utilization snapshots via nvidia-smi.
      * Runs MinerU without warmup to capture baseline latency.
      * Runs warmup and reprocesses the PDF to measure warmed latency.
      * Verifies expected output files exist.
    """

    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    with ProgressBar(
        total=8,
        desc="Validation pipeline",
        unit="step",
        enabled=show_progress,
        leave=True,
    ) as pipeline_bar:
        pipeline_bar.set_postfix({"stage": "load config"})
        config = load_config()
        pipeline_bar.update()

        pipeline_bar.set_postfix({"stage": "ensure model"})
        try:
            ensure_model_downloaded(config=config)
        except ModelDownloadError as exc:
            raise ValidationFailure(str(exc)) from exc
        pipeline_bar.update()

        pipeline_bar.set_postfix({"stage": "enforce gpu"})
        devices = _parse_cuda_devices(config.cuda_visible_devices)
        try:
            gpu_info = enforce_gpu_environment(devices=devices)
        except GPUUnavailableError as exc:
            raise ValidationFailure(str(exc)) from exc
        pipeline_bar.set_postfix(
            {
                "stage": "enforce gpu",
                "cuda": f"device {gpu_info.index}",
                "gpu": gpu_info.name,
                "backend": config.backend_name,
            }
        )
        pipeline_bar.update()

        pipeline_bar.set_postfix({"stage": "gpu snapshot (before)"})
        snapshot_before = _collect_gpu_snapshot()
        pipeline_bar.update()

        base_output_root = (
            Path(output_root).expanduser().resolve() if output_root else pdf.parent
        )
        cold_output = base_output_root / f"{pdf.stem}_cold"
        warm_output = base_output_root / f"{pdf.stem}_warm"

        with ProgressBar(
            total=1,
            desc=f"{pdf.name} cold run",
            unit="doc",
            enabled=show_progress,
            leave=False,
            position=1,
        ) as run_bar:
            pipeline_bar.set_postfix({"stage": "cold run", "output": cold_output.name})
            cold_metrics = _measure_processing_run(
                pdf,
                output_dir=cold_output,
                config=config,
                warmup=False,
            expected_files=expected_files,
            show_progress=False,
        )
            run_bar.update()
        pipeline_bar.update()

        pipeline_bar.set_postfix({"stage": "gpu warmup"})
        try:
            warmup_duration = warmup_gpu(devices[0], show_progress=show_progress)
        except GPUUnavailableError as exc:
            raise ValidationFailure(str(exc)) from exc
        pipeline_bar.update()

        with ProgressBar(
            total=1,
            desc=f"{pdf.name} warm run",
            unit="doc",
            enabled=show_progress,
            leave=False,
            position=1,
        ) as run_bar:
            pipeline_bar.set_postfix({"stage": "warm run", "output": warm_output.name})
            warm_metrics = _measure_processing_run(
                pdf,
                output_dir=warm_output,
                config=config,
                warmup=False,
            expected_files=expected_files,
            show_progress=False,
        )
            run_bar.update()
        pipeline_bar.update()

        pipeline_bar.set_postfix({"stage": "gpu snapshot (after)"})
        snapshot_after = _collect_gpu_snapshot()
        pipeline_bar.update()

    return ValidationReport(
        pdf_path=pdf,
        baseline=cold_metrics,
        warmed=warm_metrics,
        warmup_duration_seconds=warmup_duration,
        gpu_info=gpu_info,
        gpu_snapshot_before=snapshot_before,
        gpu_snapshot_after=snapshot_after,
    )


__all__ = [
    "RunMetrics",
    "ValidationFailure",
    "ValidationReport",
    "validate_pdf_processing",
]
