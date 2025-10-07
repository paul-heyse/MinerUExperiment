from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .gpu_utils import GPUUnavailableError, enforce_gpu_environment, warmup_gpu
from .mineru_config import MineruConfig, load_config, write_config
from .progress import ProgressBar

LOGGER = logging.getLogger(__name__)

EXPECTED_OUTPUT_FILES = ("markdown.md", "content_list.json", "middle.json")


class MineruInvocationError(RuntimeError):
    """Raised when the MinerU CLI call fails."""


@dataclass
class MineruProcessResult:
    success: bool
    output_dir: Path
    output_files: Dict[str, Path]
    stdout: str
    stderr: str
    error: Optional[str] = None


def _normalize_pdf_path(pdf_path: Path | str) -> Path:
    candidate = Path(pdf_path).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"PDF not found: {candidate}")
    if candidate.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file but got: {candidate}")
    return candidate.resolve()


def _resolve_output_dir(pdf_path: Path, output_dir: Optional[Path | str]) -> Path:
    if output_dir is not None:
        resolved = Path(output_dir).expanduser().resolve()
    else:
        resolved = pdf_path.parent / f"{pdf_path.stem}_mineru"
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _detect_cli() -> str:
    mineru_cli = shutil.which("mineru")
    if mineru_cli:
        return mineru_cli

    python_dir = Path(sys.executable).resolve().parent
    candidate = python_dir / "mineru"
    if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate)

    raise MineruInvocationError(
        "The 'mineru' CLI is not available. Install MinerU with `uv pip install \"mineru[all]\"`."
    )


def _parse_visible_devices(value: str) -> List[int]:
    devices: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            devices.append(int(item))
        except ValueError as exc:  # pragma: no cover - defensive path
            raise MineruInvocationError(f"Invalid CUDA device index '{item}'") from exc
    if not devices:
        devices.append(0)
    return devices


def _collect_expected_outputs(output_dir: Path, expected_files: Sequence[str]) -> Dict[str, Path]:
    results: Dict[str, Path] = {}
    for filename in expected_files:
        candidate = next(output_dir.glob(f"**/{filename}"), None)
        if candidate is not None:
            results[filename] = candidate
    return results


def _build_command(
    mineru_cli: str,
    *,
    pdf_path: Path,
    output_dir: Path,
    backend: str,
    additional_args: Sequence[str] | None,
) -> List[str]:
    command = [
        mineru_cli,
        "-p",
        str(pdf_path),
        "-o",
        str(output_dir),
        "-b",
        backend,
    ]
    if additional_args:
        command.extend(additional_args)
    return command


def process_pdf(
    pdf_path: Path | str,
    *,
    output_dir: Optional[Path | str] = None,
    config: Optional[MineruConfig] = None,
    warmup: bool = True,
    expected_files: Sequence[str] = EXPECTED_OUTPUT_FILES,
    additional_args: Optional[Sequence[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
    show_progress: bool = True,
) -> MineruProcessResult:
    """
    Invoke the MinerU CLI against a single PDF using the vLLM backend.

    Parameters
    ----------
    pdf_path:
        Path to the PDF that should be processed.
    output_dir:
        Optional directory where MinerU will place its outputs. Defaults to `<pdf>_mineru`.
    config:
        MineruConfig instance to use. If omitted, the config is loaded and written back to disk.
    warmup:
        Whether to run the GPU warmup routine before invoking the CLI.
    expected_files:
        Names of files that should exist after a successful run.
    additional_args:
        Extra command-line arguments to pass to the MinerU CLI.
    extra_env:
        Additional environment variables for the subprocess.
    """

    total_steps = 5 + (1 if warmup else 0)
    with ProgressBar(
        total=total_steps,
        desc="MinerU process",
        unit="step",
        enabled=show_progress,
        leave=True,
    ) as progress:
        progress.set_postfix({"stage": "prepare io"})
        pdf = _normalize_pdf_path(pdf_path)
        output_directory = _resolve_output_dir(pdf, output_dir)
        progress.update()

        progress.set_postfix({"stage": "load config"})
        active_config = config or load_config()
        progress.update()

        devices = _parse_visible_devices(active_config.cuda_visible_devices)
        device_str = ",".join(str(index) for index in devices)
        progress.set_postfix(
            {
                "stage": "verify cuda",
                "devices": device_str or "0",
                "backend": active_config.backend_name,
            }
        )
        try:
            enforce_gpu_environment(devices=devices, require_specific_gpu="RTX 5090")
        except GPUUnavailableError as exc:
            progress.set_postfix({"stage": "cuda error", "detail": str(exc)})
            raise MineruInvocationError(f"GPU availability check failed: {exc}") from exc
        progress.update()

        if warmup:
            progress.set_postfix({"stage": "gpu warmup", "device": device_str or "0"})
            try:
                warmup_gpu(devices[0], show_progress=show_progress)
            except GPUUnavailableError as exc:
                progress.set_postfix({"stage": "warmup skipped", "detail": str(exc)})
                LOGGER.warning("Skipping GPU warmup: %s", exc)
            progress.update()

        mineru_cli = _detect_cli()
        command = _build_command(
            mineru_cli,
            pdf_path=pdf,
            output_dir=output_directory,
            backend=active_config.backend_name,
            additional_args=list(additional_args) if additional_args else None,
        )

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device_str
        env.setdefault("MINERU_BACKEND", active_config.backend_name)
        if active_config.model_path:
            env.setdefault("MINERU_MODEL_PATH", str(active_config.model_path))
        env.setdefault("MINERU_MODEL_SOURCE", active_config.model_source)
        if extra_env:
            env.update(extra_env)

        progress.set_postfix({"stage": "run mineru", "cli": mineru_cli})
        LOGGER.info("Running MinerU: %s", " ".join(command))
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        progress.update()

        stdout = process.stdout or ""
        stderr = process.stderr or ""
        success = process.returncode == 0

        if success:
            progress.set_postfix({"stage": "collect outputs"})
            output_files = _collect_expected_outputs(output_directory, expected_files)
            progress.update()
            if len(output_files) < len(expected_files):
                missing = set(expected_files) - set(output_files)
                error_message = (
                    "MinerU finished without errors but missing expected outputs: "
                    + ", ".join(sorted(missing))
                )
                LOGGER.error(error_message)
                return MineruProcessResult(
                    success=False,
                    output_dir=output_directory,
                    output_files=output_files,
                    stdout=stdout,
                    stderr=stderr,
                    error=error_message,
                )
            if config is None:
                write_config(active_config)
            return MineruProcessResult(
                success=True,
                output_dir=output_directory,
                output_files=output_files,
                stdout=stdout,
                stderr=stderr,
                error=None,
            )

        progress.set_postfix({"stage": "mineru error", "code": process.returncode})
        progress.update()

    error_detail = stderr.strip() or stdout.strip() or "unknown error"
    LOGGER.error("MinerU failed (%d): %s", process.returncode, error_detail)
    return MineruProcessResult(
        success=False,
        output_dir=output_directory,
        output_files={},
        stdout=stdout,
        stderr=stderr,
        error=error_detail,
    )
