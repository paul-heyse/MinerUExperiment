#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from MinerUExperiment.batch_processor import BatchProcessor, BatchProcessorConfig
from MinerUExperiment.performance_config import (
    apply_profile_to_config,
    list_profiles,
    validate_profile_name,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - argparse already guards but defensive
        raise argparse.ArgumentTypeError("Expected integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be >= 1")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected float") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _non_negative_float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return parsed


def _percent(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected percentage") from exc
    if not (0 < parsed <= 100):
        raise argparse.ArgumentTypeError("Percentage must be between 0 and 100")
    return parsed


def _env_kv(value: str) -> Dict[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Environment overrides must be KEY=VALUE")
    key, val = value.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("Environment override key cannot be empty")
    return {key: val}


def _profile_help() -> str:
    lines = ["Performance profiles:"]
    for profile in list_profiles():
        temp_info = (
            f", temp≤{profile.gpu_pause_temperature_c:.0f}°C"
            if profile.gpu_pause_temperature_c is not None
            else ""
        )
        lines.append(
            f"  {profile.name:<10} - {profile.description} (workers={profile.workers.worker_count}, "
            f"gpu_mem={profile.vllm.gpu_memory_utilization:.2f}, "
            f"throttle util={profile.gpu_pause_utilization * 100:.0f}%, "
            f"mem={profile.gpu_pause_memory * 100:.0f}%{temp_info})"
        )
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch process PDFs with MinerU (vLLM backend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_profile_help(),
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("PDFsToProcess"),
        help="Directory containing PDFs to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("MDFilesCreated"),
        help="Directory to write processed outputs",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=None,
        help="Number of worker processes (default auto-detects CPU cores - 2)",
    )
    parser.add_argument(
        "--max-retries",
        type=_positive_int,
        default=3,
        help="Maximum retry attempts per PDF",
    )
    parser.add_argument(
        "--retry-delay",
        type=_non_negative_float,
        default=10.0,
        help="Seconds to wait between retries",
    )
    parser.add_argument(
        "--poll-interval",
        type=_non_negative_float,
        default=2.0,
        help="Seconds workers wait before re-checking for work",
    )
    parser.add_argument(
        "--progress-interval",
        type=_non_negative_float,
        default=5.0,
        help="Seconds between progress updates",
    )
    parser.add_argument(
        "--backend",
        default="vlm-vllm-engine",
        help="MinerU backend to use",
    )
    parser.add_argument(
        "--mineru-cli",
        default="mineru",
        help="MinerU executable to invoke",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=None,
        help="Additional argument to pass through to MinerU (repeatable)",
    )
    parser.add_argument(
        "--env",
        action="append",
        type=_env_kv,
        default=None,
        help="Environment override KEY=VALUE (repeatable)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable periodic progress logging",
    )
    parser.add_argument(
        "--profile",
        choices=[profile.name for profile in list_profiles()],
        default="balanced",
        help="Performance tuning profile to apply",
    )
    parser.add_argument(
        "--gpu-memory-throttle",
        type=_percent,
        default=None,
        help="GPU memory percent to pause dispatch (profile default)",
    )
    parser.add_argument(
        "--gpu-memory-resume",
        type=_percent,
        default=None,
        help="GPU memory percent to resume dispatch (profile default)",
    )
    parser.add_argument(
        "--gpu-util-throttle",
        type=_percent,
        default=None,
        help="GPU utilization percent to pause dispatch (profile default)",
    )
    parser.add_argument(
        "--gpu-util-resume",
        type=_percent,
        default=None,
        help="GPU utilization percent to resume dispatch (profile default)",
    )
    parser.add_argument(
        "--gpu-temp-throttle",
        type=_positive_float,
        default=None,
        help="GPU temperature (°C) to pause dispatch (profile default)",
    )
    parser.add_argument(
        "--gpu-temp-resume",
        type=_positive_float,
        default=None,
        help="GPU temperature (°C) to resume dispatch (profile default)",
    )
    parser.add_argument(
        "--gpu-monitor-interval",
        type=_positive_float,
        default=None,
        help="Seconds between GPU telemetry samples (profile default)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmark mode with enhanced metrics output",
    )
    parser.add_argument(
        "--profile-info",
        action="store_true",
        help="Print profile descriptions and exit",
    )

    return parser.parse_args(argv)


def merge_env(overrides: List[Dict[str, str]] | None) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    if overrides:
        for item in overrides:
            merged.update(item)
    return merged


def build_config(args: argparse.Namespace) -> BatchProcessorConfig:
    input_dir = args.input_dir
    output_dir = args.output_dir
    workers = args.workers if args.workers is not None else max(1, (os.cpu_count() or 4) - 2)
    extra_args = tuple(args.extra_arg or [])
    env_overrides = merge_env(args.env)

    config_values: Dict[str, object] = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "workers": workers,
        "poll_interval": max(args.poll_interval, 0.1),
        "max_retries": args.max_retries,
        "retry_delay": args.retry_delay,
        "progress_interval": max(args.progress_interval, 0.1),
        "mineru_cli": args.mineru_cli,
        "mineru_backend": args.backend,
        "mineru_extra_args": extra_args,
        "env_overrides": env_overrides,
        "log_progress": not args.no_progress,
        "memory_pause_threshold": 0.80,
        "memory_resume_threshold": 0.70,
        "cpu_pause_threshold": 0.90,
        "resource_monitor_interval": 5.0,
        "profile": args.profile,
        "max_workers": 14,
        "gpu_memory_utilization": 0.90,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "max_model_len": 16384,
        "block_size": 16,
        "swap_space_mb": 8192,
        "dtype": "bfloat16",
        "io_buffer_size": 10 * 1024 * 1024,
        "enable_cpu_affinity": True,
        "reserved_system_cores": 2,
        "worker_memory_limit_gb": 12.0,
        "worker_niceness": 5,
        "omp_threads": 1,
        "mkl_threads": 1,
        "cpu_affinity_plan": {},
        "benchmark": args.benchmark,
        "performance_report_path": output_dir / "performance_report.json",
    }

    profile = validate_profile_name(args.profile)
    apply_profile_to_config(
        config=config_values,
        profile=profile,
        workers_override=args.workers,
    )

    if args.gpu_memory_throttle is not None:
        config_values["gpu_pause_memory_threshold"] = args.gpu_memory_throttle / 100.0
    if args.gpu_memory_resume is not None:
        config_values["gpu_resume_memory_threshold"] = args.gpu_memory_resume / 100.0
    if args.gpu_util_throttle is not None:
        config_values["gpu_pause_utilization_threshold"] = args.gpu_util_throttle / 100.0
    if args.gpu_util_resume is not None:
        config_values["gpu_resume_utilization_threshold"] = args.gpu_util_resume / 100.0
    if args.gpu_temp_throttle is not None:
        config_values["gpu_pause_temperature_c"] = args.gpu_temp_throttle
    if args.gpu_temp_resume is not None:
        config_values["gpu_resume_temperature_c"] = args.gpu_temp_resume
    if args.gpu_monitor_interval is not None:
        config_values["gpu_monitor_interval"] = max(args.gpu_monitor_interval, 0.1)

    config_values["mineru_extra_args"] = tuple(config_values["mineru_extra_args"])
    config_values["env_overrides"] = dict(config_values["env_overrides"])

    return BatchProcessorConfig(**config_values)  # type: ignore[arg-type]


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    if args.profile_info:
        print(_profile_help())
        return 0
    configure_logging(args.log_level)

    try:
        config = build_config(args)
        processor = BatchProcessor(config)
        summary = processor.run()
    except Exception as exc:
        logging.exception("Batch processing failed: %s", exc)
        return 1

    logging.info(
        "Summary: processed=%d success=%d failed=%d duration=%.2fs",
        summary.processed,
        summary.succeeded,
        summary.failed,
        summary.duration_seconds,
    )
    return 0 if summary.failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
