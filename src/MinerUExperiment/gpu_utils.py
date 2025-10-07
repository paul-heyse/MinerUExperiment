from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

LOGGER = logging.getLogger(__name__)


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
    for _ in range(iterations):
        _ = torch.matmul(dummy_input, dummy_weight)
        torch.cuda.synchronize(device)

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


def enforce_gpu_environment(
    *,
    devices: Iterable[int] = (0,),
    require_specific_gpu: Optional[str] = "RTX 5090",
) -> GPUInfo:
    device_list = list(devices)
    ensure_cuda_visible_devices(device_list)
    gpu_info = verify_gpu(require_specific_gpu or "", device_list[0])
    return gpu_info
