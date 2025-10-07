from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
    io_wait_percent: float


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

        self._active: Dict[tuple[str, str], tuple[float, int, float]] = {}
        self._records: List[PDFMetric] = []
        self._gpu_samples: List[GPUSample] = []
        self._cpu_samples: List[CPUSample] = []

    # ------------------------------------------------------------------

    def start(self) -> None:
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
        cpu_times = psutil.cpu_times_percent(interval=None)
        io_wait = getattr(cpu_times, "iowait", 0.0)
        sample = CPUSample(
            timestamp=time.time(),
            per_core_percent=per_core,
            system_memory_percent=mem.percent,
            io_wait_percent=float(io_wait),
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
        avg_system_mem = (
            sum(sample["system_memory_percent"] for sample in cpu_samples) / len(cpu_samples)
            if cpu_samples
            else 0.0
        )
        max_system_mem = max((sample["system_memory_percent"] for sample in cpu_samples), default=0.0)
        avg_io_wait = (
            sum(sample["io_wait_percent"] for sample in cpu_samples) / len(cpu_samples)
            if cpu_samples
            else 0.0
        )
        avg_cpu_per_core: List[float] = []
        if cpu_samples:
            core_count = len(cpu_samples[0]["per_core_percent"])
            for idx in range(core_count):
                avg_cpu_per_core.append(
                    sum(sample["per_core_percent"][idx] for sample in cpu_samples) / len(cpu_samples)
                )

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
                "average_system_memory_percent": avg_system_mem,
                "max_system_memory_percent": max_system_mem,
                "average_cpu_utilization_per_core": avg_cpu_per_core,
                "average_io_wait_percent": avg_io_wait,
            },
        }

        if orjson is not None:
            report_path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2) + b"\n")
        else:
            with report_path.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")

        LOGGER.info("Performance report written to %s", report_path)
        return report_path


def load_performance_report(path: Path) -> Dict[str, object]:
    with path.open("rb") as handle:
        data = handle.read()
    if not data:
        raise ValueError(f"Performance report {path} is empty")
    if orjson is not None:
        return orjson.loads(data)
    return json.loads(data.decode("utf-8"))


def compare_reports(
    report_paths: Sequence[Path],
    *,
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    if not report_paths:
        raise ValueError("At least one report path must be provided")

    comparisons: List[Dict[str, object]] = []
    for path in report_paths:
        report = load_performance_report(path)
        summary = report.get("summary", {}) if isinstance(report, dict) else {}
        throughput = float(summary.get("throughput_pdfs_per_hour", 0.0) or 0.0)
        comparisons.append(
            {
                "path": str(path),
                "profile": report.get("profile"),
                "throughput_pdfs_per_hour": throughput,
                "avg_seconds_per_pdf": float(summary.get("avg_seconds_per_pdf", 0.0) or 0.0),
                "succeeded": int(summary.get("succeeded", 0) or 0),
                "duration_seconds": float(summary.get("duration_seconds", 0.0) or 0.0),
            }
        )

    baseline = comparisons[0]
    baseline_throughput = baseline["throughput_pdfs_per_hour"] or 0.0
    for entry in comparisons:
        change_pct: Optional[float]
        if baseline_throughput:
            change_pct = ((entry["throughput_pdfs_per_hour"] / baseline_throughput) - 1.0) * 100.0
        else:
            change_pct = None
        entry["throughput_change_pct"] = change_pct

    best = max(comparisons, key=lambda item: item["throughput_pdfs_per_hour"])
    result: Dict[str, object] = {
        "baseline": baseline,
        "best": best,
        "comparisons": sorted(
            comparisons,
            key=lambda item: item["throughput_pdfs_per_hour"],
            reverse=True,
        ),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if orjson is not None:
            output_path.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2) + b"\n")
        else:
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
                handle.write("\n")

    return result


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
    "load_performance_report",
    "compare_reports",
]
