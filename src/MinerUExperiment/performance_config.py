from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Optional, Tuple

LOGGER_NAME = "MinerUExperiment.performance_config"


@dataclass(frozen=True)
class PipelineProfile:
    worker_count: int
    max_workers: int
    memory_per_worker_gb: float
    reserved_system_cores: int
    niceness: int
    omp_threads: int
    mkl_threads: int
    enable_cpu_affinity: bool = True
    mineru_extra_args: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PerformanceProfile:
    name: str
    description: str
    workers: PipelineProfile
    mineru_extra_args: Tuple[str, ...] = ()
    env_overrides: Mapping[str, str] = field(default_factory=dict)
    memory_pause_threshold: float = 0.90
    memory_resume_threshold: float = 0.75
    cpu_pause_threshold: float = 0.92


def _default_env() -> Dict[str, str]:
    return {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.8",
    }


PROFILE_REGISTRY: Dict[str, PerformanceProfile] = {
    "throughput": PerformanceProfile(
        name="throughput",
        description="Maximize PDFs/hour with aggressive batching and full GPU utilization.",
        workers=PipelineProfile(
            worker_count=14,
            max_workers=14,
            memory_per_worker_gb=11.0,
            reserved_system_cores=2,
            niceness=10,
            omp_threads=1,
            mkl_threads=1,
            enable_cpu_affinity=True,
            mineru_extra_args=("--batch-mode", "aggressive"),
        ),
        env_overrides=_default_env(),
        memory_pause_threshold=0.92,
        memory_resume_threshold=0.80,
        cpu_pause_threshold=0.95,
    ),
    "balanced": PerformanceProfile(
        name="balanced",
        description="Balanced throughput and stability with moderate GPU utilization.",
        workers=PipelineProfile(
            worker_count=12,
            max_workers=14,
            memory_per_worker_gb=10.5,
            reserved_system_cores=2,
            niceness=5,
            omp_threads=1,
            mkl_threads=1,
            mineru_extra_args=("--batch-mode", "balanced"),
        ),
        env_overrides=_default_env(),
        memory_pause_threshold=0.88,
        memory_resume_threshold=0.72,
        cpu_pause_threshold=0.92,
    ),
    "latency": PerformanceProfile(
        name="latency",
        description="Minimize per-PDF latency with fewer workers and conservative GPU usage.",
        workers=PipelineProfile(
            worker_count=6,
            max_workers=8,
            memory_per_worker_gb=12.0,
            reserved_system_cores=2,
            niceness=0,
            omp_threads=1,
            mkl_threads=1,
            mineru_extra_args=("--batch-mode", "low-latency"),
        ),
        env_overrides={
            **_default_env(),
            "MINERU_PIPELINE_PREFETCH": "disabled",
        },
        memory_pause_threshold=0.85,
        memory_resume_threshold=0.70,
        cpu_pause_threshold=0.90,
    ),
}

DEFAULT_PROFILE = PROFILE_REGISTRY["balanced"]


def list_profiles() -> List[PerformanceProfile]:
    return list(PROFILE_REGISTRY.values())


def validate_profile_name(name: str) -> PerformanceProfile:
    key = name.strip().lower()
    if key not in PROFILE_REGISTRY:
        available = ", ".join(sorted(PROFILE_REGISTRY))
        raise KeyError(f"Unknown performance profile '{name}'. Available: {available}")
    return PROFILE_REGISTRY[key]


def plan_cpu_affinity(
    worker_count: int,
    *,
    reserved_cores: int = 2,
    total_cores: Optional[int] = None,
) -> Dict[int, List[int]]:
    total = total_cores if total_cores is not None else (os.cpu_count() or 0)
    if total <= reserved_cores or worker_count <= 0:
        return {}

    usable_cores = list(range(reserved_cores, total))
    if not usable_cores:
        return {}

    plan: Dict[int, List[int]] = {}
    for worker_idx in range(worker_count):
        core_idx = worker_idx % len(usable_cores)
        plan[worker_idx] = [usable_cores[core_idx]]
    return plan


def apply_profile_to_config(
    *,
    config: MutableMapping[str, object],
    profile: PerformanceProfile,
    workers_override: Optional[int] = None,
) -> None:
    worker_settings = profile.workers
    worker_count = workers_override if workers_override is not None else worker_settings.worker_count
    worker_count = max(1, min(worker_count, worker_settings.max_workers))

    config["profile"] = profile.name
    config["workers"] = worker_count
    config["max_workers"] = worker_settings.max_workers
    config["reserved_system_cores"] = worker_settings.reserved_system_cores
    config["worker_memory_limit_gb"] = worker_settings.memory_per_worker_gb
    config["worker_niceness"] = worker_settings.niceness
    config["omp_threads"] = worker_settings.omp_threads
    config["mkl_threads"] = worker_settings.mkl_threads
    config["enable_cpu_affinity"] = worker_settings.enable_cpu_affinity
    config["memory_pause_threshold"] = profile.memory_pause_threshold
    config["memory_resume_threshold"] = profile.memory_resume_threshold
    config["cpu_pause_threshold"] = profile.cpu_pause_threshold
    combined_args = tuple(config.get("mineru_extra_args", tuple()))
    combined_args += worker_settings.mineru_extra_args
    combined_args += profile.mineru_extra_args
    config["mineru_extra_args"] = combined_args

    env_overrides: Dict[str, str] = dict(config.get("env_overrides", {}))
    for key, value in profile.env_overrides.items():
        env_overrides.setdefault(key, value)
    env_overrides.setdefault("OMP_NUM_THREADS", str(worker_settings.omp_threads))
    env_overrides.setdefault("MKL_NUM_THREADS", str(worker_settings.mkl_threads))
    config["env_overrides"] = env_overrides

    affinity_plan = plan_cpu_affinity(
        worker_count,
        reserved_cores=worker_settings.reserved_system_cores,
    )
    config["cpu_affinity_plan"] = affinity_plan
