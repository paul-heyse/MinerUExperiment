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
        worker_memory_limit_gb=0.1,
        dtype="float16",
    )


def test_resource_monitor_throttles_and_resumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    processor = BatchProcessor(config)

    mem_samples = [92.0, 65.0, 58.0]
    cpu_samples = [95.0, 72.0, 35.0]

    def fake_virtual_memory() -> SimpleNamespace:
        percent = mem_samples.pop(0) if mem_samples else 58.0
        return SimpleNamespace(percent=percent)

    def fake_cpu_percent(interval=None) -> float:
        return cpu_samples.pop(0) if cpu_samples else 35.0

    monkeypatch.setattr("MinerUExperiment.batch_processor.psutil.virtual_memory", fake_virtual_memory)
    monkeypatch.setattr("MinerUExperiment.batch_processor.psutil.cpu_percent", fake_cpu_percent)

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
