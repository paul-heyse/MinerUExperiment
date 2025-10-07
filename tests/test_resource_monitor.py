import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from MinerUExperiment.batch_processor import BatchProcessor, BatchProcessorConfig
from MinerUExperiment.gpu_utils import GPUTelemetry


def _make_config(tmp_path: Path) -> BatchProcessorConfig:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()
    return BatchProcessorConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        workers=1,
        poll_interval=0.1,
        max_retries=1,
        retry_delay=0.1,
        progress_interval=0.1,
        mineru_cli="mineru",
        mineru_backend="test",
        mineru_extra_args=tuple(),
        env_overrides={},
        log_progress=False,
        resource_monitor_interval=0.1,
        gpu_monitor_interval=0.1,
        worker_memory_limit_gb=0.1,
        dtype="float16",
    )


def test_gpu_saturation_triggers_and_clears_throttle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    processor = BatchProcessor(config)

    samples = [
        GPUTelemetry(index=0, utilization_percent=99.0, memory_used_mb=23000.0, memory_total_mb=24576.0, temperature_c=88.0),
        GPUTelemetry(index=0, utilization_percent=82.0, memory_used_mb=15000.0, memory_total_mb=24576.0, temperature_c=75.0),
        GPUTelemetry(index=0, utilization_percent=80.0, memory_used_mb=14000.0, memory_total_mb=24576.0, temperature_c=74.0),
    ]

    def fake_sample() -> GPUTelemetry:
        if samples:
            return samples.pop(0)
        return GPUTelemetry(
            index=0,
            utilization_percent=78.0,
            memory_used_mb=13000.0,
            memory_total_mb=24576.0,
            temperature_c=72.0,
        )

    monkeypatch.setattr("MinerUExperiment.batch_processor.sample_gpu_telemetry", lambda: fake_sample())
    monkeypatch.setattr(
        "MinerUExperiment.batch_processor.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=48.0),
    )
    monkeypatch.setattr(
        "MinerUExperiment.batch_processor.psutil.cpu_percent",
        lambda interval=None: 22.0,
    )

    time_box = {"value": 0.0}

    def fake_time() -> float:
        return time_box["value"]

    def fake_sleep(interval: float) -> None:
        time_box["value"] += interval
        if time_box["value"] >= 0.35:
            processor._shutdown_event.set()

    monkeypatch.setattr("MinerUExperiment.batch_processor.time.time", fake_time)
    monkeypatch.setattr("MinerUExperiment.batch_processor.time.sleep", fake_sleep)

    set_calls: list[str] = []
    original_set = processor._throttle_event.set
    original_clear = processor._throttle_event.clear

    def record_set() -> None:
        set_calls.append("set")
        original_set()

    def record_clear() -> None:
        set_calls.append("clear")
        original_clear()

    monkeypatch.setattr(processor._throttle_event, "set", record_set)
    monkeypatch.setattr(processor._throttle_event, "clear", record_clear)

    thread = threading.Thread(target=processor._resource_monitor_loop)
    thread.start()
    thread.join(timeout=1.0)

    assert "set" in set_calls
    assert "clear" in set_calls[set_calls.index("set") + 1 :]
    assert not processor._throttle_event.is_set()
