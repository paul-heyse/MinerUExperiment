import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from MinerUExperiment.batch_processor import (
    BatchProcessorConfig,
    _mineru_command,
    _worker_env,
)


def _make_config(tmp_path: Path) -> BatchProcessorConfig:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()
    return BatchProcessorConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        workers=3,
        poll_interval=0.5,
        max_retries=2,
        retry_delay=1.0,
        progress_interval=1.5,
        mineru_cli="mineru",
        mineru_backend="backend",
        mineru_extra_args=("--foo", "bar"),
        env_overrides={"CUSTOM": "1", "OMP_NUM_THREADS": "16"},
        log_progress=True,
        memory_pause_threshold=0.85,
        memory_resume_threshold=0.75,
        cpu_pause_threshold=0.93,
        resource_monitor_interval=1.0,
        gpu_memory_utilization=0.88,
        tensor_parallel_size=2,
        data_parallel_size=3,
        max_model_len=2048,
        block_size=8,
        swap_space_mb=4096,
        dtype="float16",
        worker_memory_limit_gb=8.0,
    )


def test_worker_env_populates_defaults_and_preserves_overrides(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    env = _worker_env(config)

    assert env["CUSTOM"] == "1"
    assert env["OMP_NUM_THREADS"] == "16"
    assert env["MKL_NUM_THREADS"] == str(config.mkl_threads)
    assert env["MINERU_VLLM_GPU_MEMORY_UTILIZATION"] == f"{config.gpu_memory_utilization:.2f}"
    assert env["MINERU_VLLM_TENSOR_PARALLEL_SIZE"] == str(config.tensor_parallel_size)
    assert env["MINERU_VLLM_DATA_PARALLEL_SIZE"] == str(config.data_parallel_size)
    assert env["MINERU_VLLM_MAX_MODEL_LEN"] == str(config.max_model_len)
    assert env["MINERU_VLLM_BLOCK_SIZE"] == str(config.block_size)
    assert env["MINERU_VLLM_SWAP_SPACE_MB"] == str(config.swap_space_mb)
    assert env["MINERU_VLLM_DTYPE"] == config.dtype


def test_mineru_command_includes_extra_args(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    pdf_path = config.input_dir / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")
    output_dir = config.output_dir / "doc"
    command = _mineru_command(
        cli=config.mineru_cli,
        pdf_path=pdf_path,
        output_dir=output_dir,
        backend=config.mineru_backend,
        extra_args=config.mineru_extra_args,
    )

    assert command[:7] == [
        config.mineru_cli,
        "-p",
        str(pdf_path),
        "-o",
        str(output_dir),
        "-b",
        config.mineru_backend,
    ]
    assert list(config.mineru_extra_args) == command[7:]


def test_worker_env_isolated_per_call(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setenv("MINERU_VLLM_GPU_MEMORY_UTILIZATION", "0.11")
    env = _worker_env(config)

    assert env["MINERU_VLLM_GPU_MEMORY_UTILIZATION"] == "0.11"
    os.environ.pop("MINERU_VLLM_GPU_MEMORY_UTILIZATION", None)
    env2 = _worker_env(config)
    assert env2["MINERU_VLLM_GPU_MEMORY_UTILIZATION"] == f"{config.gpu_memory_utilization:.2f}"
