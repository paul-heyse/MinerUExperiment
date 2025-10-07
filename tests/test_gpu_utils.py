import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import MinerUExperiment.gpu_utils as gpu_utils


def test_sample_gpu_telemetry_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("pynvml", None)

    class DummyUtil:
        gpu = 76

    class DummyMem:
        used = 1024 * 1024 * 12
        total = 1024 * 1024 * 24

    class DummyNVML:
        NVML_TEMPERATURE_GPU = 0

        @staticmethod
        def nvmlInit() -> None:
            return None

        @staticmethod
        def nvmlShutdown() -> None:
            return None

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index: int) -> int:
            return index

        @staticmethod
        def nvmlDeviceGetUtilizationRates(handle: int) -> DummyUtil:
            return DummyUtil()

        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle: int) -> DummyMem:
            return DummyMem()

        @staticmethod
        def nvmlDeviceGetTemperature(handle: int, sensor: int) -> float:
            return 71.5

    monkeypatch.setitem(sys.modules, "pynvml", DummyNVML)

    telemetry = gpu_utils.sample_gpu_telemetry(0)
    assert telemetry is not None
    assert telemetry.utilization_percent == pytest.approx(76.0)
    assert telemetry.memory_used_mb == pytest.approx(12.0)
    assert telemetry.memory_total_mb == pytest.approx(24.0)
    assert telemetry.memory_percent == pytest.approx(50.0)
    assert telemetry.temperature_c == pytest.approx(71.5)

    monkeypatch.delitem(sys.modules, "pynvml", raising=False)


def test_sample_gpu_telemetry_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("pynvml", None)

    monkeypatch.setattr(gpu_utils.shutil, "which", lambda command: "/usr/bin/nvidia-smi")

    def fake_run(cmd, capture_output, text, check, timeout):
        assert "--query-gpu" in cmd[1]
        return SimpleNamespace(stdout="97, 22000, 24576, 83\n")

    monkeypatch.setattr(gpu_utils.subprocess, "run", fake_run)

    telemetry = gpu_utils.sample_gpu_telemetry(0)
    assert telemetry is not None
    assert telemetry.utilization_percent == pytest.approx(97.0)
    assert telemetry.memory_used_mb == pytest.approx(22000.0)
    assert telemetry.memory_total_mb == pytest.approx(24576.0)
    assert telemetry.memory_percent == pytest.approx(22000.0 / 24576.0 * 100.0)
    assert telemetry.temperature_c == pytest.approx(83.0)


def test_sample_gpu_telemetry_none(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("pynvml", None)
    monkeypatch.setattr(gpu_utils.shutil, "which", lambda command: None)

    telemetry = gpu_utils.sample_gpu_telemetry(0)
    assert telemetry is None
