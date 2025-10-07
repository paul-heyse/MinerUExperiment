from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

from .progress import ProgressBar

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUTelemetry:
    """Current utilization snapshot for a GPU."""

    index: int
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: Optional[float] = None

    @property
    def memory_percent(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100.0


class GPUUnavailableError(RuntimeError):
    """Raised when the requested GPU cannot be used."""


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_memory_mb: int
    compute_capability: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "index": str(self.index),
            "name": self.name,
            "total_memory_mb": str(self.total_memory_mb),
            "compute_capability": self.compute_capability,
        }


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise GPUUnavailableError(
            "PyTorch is required for GPU operations but is not installed. "
            "Install torch with CUDA support to continue."
        ) from exc
    return torch


def ensure_cuda_visible_devices(devices: Sequence[int]) -> str:
    device_str = ",".join(str(device) for device in devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str
    LOGGER.debug("Set CUDA_VISIBLE_DEVICES=%s", device_str)
    return device_str


def get_gpu_info(device_index: int = 0) -> GPUInfo:
    torch = _import_torch()
    if not torch.cuda.is_available():
        raise GPUUnavailableError("CUDA is not available. Verify NVIDIA drivers and CUDA toolkit.")

    device_count = torch.cuda.device_count()
    if device_index >= device_count:
        raise GPUUnavailableError(
            f"Requested GPU index {device_index} but only {device_count} device(s) detected."
        )

    properties = torch.cuda.get_device_properties(device_index)
    compute_capability = f"{properties.major}.{properties.minor}"
    total_memory_mb = int(properties.total_memory / (1024 * 1024))
    name = properties.name

    gpu_info = GPUInfo(
        index=device_index,
        name=name,
        total_memory_mb=total_memory_mb,
        compute_capability=compute_capability,
    )
    LOGGER.info(
        "Detected GPU %s (index=%d, memory=%d MB, compute capability=%s)",
        gpu_info.name,
        gpu_info.index,
        gpu_info.total_memory_mb,
        gpu_info.compute_capability,
    )
    return gpu_info


def verify_gpu(required_name_fragment: str = "RTX 5090", device_index: int = 0) -> GPUInfo:
    gpu_info = get_gpu_info(device_index)
    if required_name_fragment.lower() not in gpu_info.name.lower():
        raise GPUUnavailableError(
            f"GPU '{gpu_info.name}' does not match required device '{required_name_fragment}'."
        )
    return gpu_info


def warmup_gpu(
    device_index: int = 0,
    *,
    iterations: int = 3,
    tensor_size: int = 262_144,
    show_progress: bool = True,
) -> float:
    torch = _import_torch()
    if not torch.cuda.is_available():
        raise GPUUnavailableError("CUDA is not available; cannot warm up GPU.")

    device = torch.device(f"cuda:{device_index}")
    matrix_dim = int(tensor_size ** 0.5)
    matrix_dim = max(matrix_dim, 512)
    dummy_input = torch.randn((matrix_dim, matrix_dim), device=device)
    dummy_weight = torch.randn((matrix_dim, matrix_dim), device=device)

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with ProgressBar(
        total=iterations,
        enabled=show_progress,
        leave=False,
        desc=f"GPU warmup cuda:{device_index}",
        unit="iter",
    ) as bar:
        for iteration in range(1, iterations + 1):
            iteration_start = time.perf_counter()
            _ = torch.matmul(dummy_input, dummy_weight)
            torch.cuda.synchronize(device)
            bar.update(1)
            elapsed_iter = time.perf_counter() - iteration_start
            bar.set_postfix({"iter_s": f"{elapsed_iter:.2f}"})

    elapsed = time.perf_counter() - start
    LOGGER.info(
        "Completed GPU warmup on device %s in %.2f seconds (%d iterations).",
        device_index,
        elapsed,
        iterations,
    )
    return elapsed


def is_gpu_available(device_index: int = 0) -> bool:
    try:
        get_gpu_info(device_index)
        return True
    except GPUUnavailableError as exc:
        LOGGER.warning("GPU check failed: %s", exc)
        return False


def _query_nvml(device_index: int) -> Optional[GPUTelemetry]:  # pragma: no cover - requires NVML
    try:
        import pynvml
    except ModuleNotFoundError:
        return None

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        LOGGER.debug("NVML init failed: %s", exc)
        return None

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        try:
            temperature = float(
                pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            temperature = None

        return GPUTelemetry(
            index=device_index,
            utilization_percent=float(utilization.gpu),
            memory_used_mb=float(memory.used) / (1024 * 1024),
            memory_total_mb=float(memory.total) / (1024 * 1024),
            temperature_c=temperature,
        )
    except Exception as exc:
        LOGGER.debug("NVML telemetry query failed: %s", exc)
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _query_nvidia_smi(device_index: int) -> Optional[GPUTelemetry]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None

    query = [
        executable,
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
        f"--id={device_index}",
    ]

    try:
        result = subprocess.run(
            query,
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None

    line = result.stdout.strip().splitlines()
    if not line:
        return None

    parts = [item.strip() for item in line[0].split(",")]
    try:
        util = float(parts[0])
        mem_used = float(parts[1])
        mem_total = float(parts[2]) if len(parts) > 2 else 0.0
        temp = float(parts[3]) if len(parts) > 3 else None
    except (ValueError, IndexError):
        return None

    if mem_total <= 0:
        # Attempt to derive total memory from PyTorch if possible.
        try:
            info = get_gpu_info(device_index)
            mem_total = float(info.total_memory_mb)
        except GPUUnavailableError:
            mem_total = 0.0

    return GPUTelemetry(
        index=device_index,
        utilization_percent=util,
        memory_used_mb=mem_used,
        memory_total_mb=mem_total,
        temperature_c=temp,
    )


def sample_gpu_telemetry(device_index: int = 0) -> Optional[GPUTelemetry]:
    """Return current GPU telemetry, or ``None`` if unavailable."""

    telemetry = _query_nvml(device_index)
    if telemetry is not None:
        return telemetry

    return _query_nvidia_smi(device_index)


def enforce_gpu_environment(
    *,
    devices: Iterable[int] = (0,),
    require_specific_gpu: Optional[str] = "RTX 5090",
) -> GPUInfo:
    device_list = list(devices)
    ensure_cuda_visible_devices(device_list)
    gpu_info = verify_gpu(require_specific_gpu or "", device_list[0])
    return gpu_info
