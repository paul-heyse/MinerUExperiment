from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from types import SimpleNamespace
from typing import Any, Dict

import pytest

import MinerUExperiment.mineru_config as mineru_config
import MinerUExperiment.mineru_runner as mineru_runner
from MinerUExperiment import (
    MineruConfig,
    MineruProcessResult,
    ensure_model_downloaded,
    load_config,
    ping,
    write_config,
)
from MinerUExperiment.performance_config import apply_profile_to_config, validate_profile_name


def test_ping() -> None:
    assert ping() == "pong"


def test_load_and_write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "mineru.json"
    monkeypatch.setattr(mineru_config, "CONFIG_PATH", config_path, raising=False)

    config = load_config()
    assert config.backend_name == "vlm-vllm-engine"

    config.set_visible_devices([0])
    config.update_vllm_settings(data_parallel_size=2)
    write_config(config)

    stored = load_config()
    assert stored.cuda_visible_devices == "0"
    assert stored.vllm_settings.data_parallel_size == 2


def test_ensure_model_downloaded_updates_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "mineru.json"
    cache_path = tmp_path / "cache"
    model_dir = cache_path / "MinerU2.5-2509-1.2B"
    model_dir.mkdir(parents=True)

    monkeypatch.setenv("MINERU_MODEL_SOURCE", "huggingface")
    monkeypatch.setattr(mineru_config, "CONFIG_PATH", config_path, raising=False)
    monkeypatch.setattr(mineru_config, "DEFAULT_MODEL_CACHE", cache_path, raising=False)
    monkeypatch.setattr(
        mineru_config,
        "_download_from_huggingface",
        lambda repo_id, cache_dir, revision, token: model_dir,
        raising=False,
    )

    config = MineruConfig()
    path = ensure_model_downloaded(config=config)

    assert path == model_dir
    assert config.model_path == model_dir.resolve()


def test_apply_profile_to_config_updates_workers() -> None:
    profile = validate_profile_name("throughput")
    config_values = {
        "workers": 4,
        "mineru_extra_args": tuple(),
        "env_overrides": {},
        "gpu_memory_utilization": 0.90,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "max_model_len": 16384,
        "block_size": 16,
        "swap_space_mb": 8192,
        "dtype": "bfloat16",
        "reserved_system_cores": 2,
        "worker_memory_limit_gb": 12.0,
        "worker_niceness": 5,
        "omp_threads": 1,
        "mkl_threads": 1,
        "cpu_affinity_plan": {},
    }

    apply_profile_to_config(config=config_values, profile=profile, workers_override=None)

    assert config_values["workers"] == profile.workers.worker_count
    assert config_values["gpu_memory_utilization"] == profile.vllm.gpu_memory_utilization
    assert "OMP_NUM_THREADS" in config_values["env_overrides"]
    assert config_values["cpu_affinity_plan"]


def _fake_successful_run(output_dir: Path) -> SimpleNamespace:
    for filename in mineru_runner.EXPECTED_OUTPUT_FILES:
        (output_dir / filename).write_text("data", encoding="utf-8")
    return SimpleNamespace(returncode=0, stdout="ok", stderr="")


def test_process_pdf_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")

    output_dir = tmp_path / "output"

    monkeypatch.setattr(mineru_runner, "enforce_gpu_environment", lambda **_: None, raising=False)
    monkeypatch.setattr(mineru_runner, "warmup_gpu", lambda *_, **__: 0.0, raising=False)
    monkeypatch.setattr(mineru_runner, "_detect_cli", lambda: "mineru", raising=False)

    def fake_run(
        command: Any,
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        env: Dict[str, str],
    ) -> SimpleNamespace:
        _ = command, check, capture_output, text, env
        output_dir.mkdir(parents=True, exist_ok=True)
        return _fake_successful_run(output_dir)

    monkeypatch.setattr(mineru_runner.subprocess, "run", fake_run, raising=False)

    config = MineruConfig()
    config.set_visible_devices([0])

    result = mineru_runner.process_pdf(
        pdf_path,
        output_dir=output_dir,
        config=config,
        warmup=False,
        show_progress=False,
    )

    assert isinstance(result, MineruProcessResult)
    assert result.success
    for filename in mineru_runner.EXPECTED_OUTPUT_FILES:
        assert filename in result.output_files


def test_process_pdf_missing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")
    output_dir = tmp_path / "output-missing"

    monkeypatch.setattr(mineru_runner, "enforce_gpu_environment", lambda **_: None, raising=False)
    monkeypatch.setattr(mineru_runner, "warmup_gpu", lambda *_, **__: 0.0, raising=False)
    monkeypatch.setattr(mineru_runner, "_detect_cli", lambda: "mineru", raising=False)

    def fake_run(
        command: Any,
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        env: Dict[str, str],
    ) -> SimpleNamespace:
        _ = command, check, capture_output, text, env
        output_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(mineru_runner.subprocess, "run", fake_run, raising=False)

    config = MineruConfig()
    config.set_visible_devices([0])

    result = mineru_runner.process_pdf(
        pdf_path,
        output_dir=output_dir,
        config=config,
        warmup=False,
        show_progress=False,
    )

    assert not result.success
    assert result.error is not None
