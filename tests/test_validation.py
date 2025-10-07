from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List
import time

import pytest

import MinerUExperiment.validation as validation
from MinerUExperiment import MineruConfig, MineruProcessResult, ValidationFailure
from MinerUExperiment.metrics import MetricsCollector


@pytest.fixture(name="pdf_file")
def fixture_pdf_file(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")
    return pdf_path


def _make_success_result(output_dir: Path) -> MineruProcessResult:
    expected = {}
    for filename in validation.EXPECTED_OUTPUT_FILES:
        path = output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data", encoding="utf-8")
        expected[filename] = path
    return MineruProcessResult(
        success=True,
        output_dir=output_dir,
        output_files=expected,
        stdout="ok",
        stderr="",
    )


def test_validate_pdf_processing_success(
    tmp_path: Path,
    pdf_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MineruConfig()
    config.set_visible_devices([0])

    call_sequence: List[str] = []

    monkeypatch.setattr(validation, "load_config", lambda: config, raising=False)
    monkeypatch.setattr(validation, "ensure_model_downloaded", lambda config: tmp_path, raising=False)
    monkeypatch.setattr(
        validation,
        "enforce_gpu_environment",
        lambda devices: SimpleNamespace(
            index=devices[0],
            name="RTX 5090",
            total_memory_mb=24_576,
            compute_capability="9.0",
            to_dict=lambda: {
                "index": str(devices[0]),
                "name": "RTX 5090",
                "total_memory_mb": "24576",
                "compute_capability": "9.0",
            },
        ),
        raising=False,
    )
    monkeypatch.setattr(validation, "_collect_gpu_snapshot", lambda: "snapshot", raising=False)
    monkeypatch.setattr(validation, "warmup_gpu", lambda device: 1.5, raising=False)

    def fake_process_pdf(*args, **kwargs):
        call_sequence.append(kwargs.get("output_dir").name)
        return _make_success_result(kwargs["output_dir"])

    monkeypatch.setattr(validation, "process_pdf", fake_process_pdf, raising=False)

    times = iter([0.0, 3.0, 10.0, 12.0])
    monkeypatch.setattr(validation.time, "perf_counter", lambda: next(times), raising=False)

    report = validation.validate_pdf_processing(pdf_file, output_root=tmp_path)

    assert report.baseline.duration_seconds == 3.0
    assert report.warmed.duration_seconds == 2.0
    assert report.warmup_duration_seconds == 1.5
    assert report.warmed_is_faster()
    assert report.gpu_snapshot_before == "snapshot"
    assert report.gpu_snapshot_after == "snapshot"
    assert call_sequence == ["document_cold", "document_warm"]
    cold_dir = tmp_path / "document_cold"
    warm_dir = tmp_path / "document_warm"
    assert cold_dir.exists()
    assert warm_dir.exists()
    assert report.baseline.result.output_dir == cold_dir
    assert report.warmed.result.output_dir == warm_dir


def test_validate_pdf_processing_failure_raises(
    tmp_path: Path,
    pdf_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MineruConfig()
    config.set_visible_devices([0])

    monkeypatch.setattr(validation, "load_config", lambda: config, raising=False)
    monkeypatch.setattr(validation, "ensure_model_downloaded", lambda config: tmp_path, raising=False)
    monkeypatch.setattr(
        validation,
        "enforce_gpu_environment",
        lambda devices: SimpleNamespace(
            index=devices[0],
            name="RTX 5090",
            total_memory_mb=24_576,
            compute_capability="9.0",
            to_dict=lambda: {
                "index": str(devices[0]),
                "name": "RTX 5090",
                "total_memory_mb": "24576",
                "compute_capability": "9.0",
            },
        ),
        raising=False,
    )
    monkeypatch.setattr(validation, "_collect_gpu_snapshot", lambda: None, raising=False)
    monkeypatch.setattr(validation, "warmup_gpu", lambda device: 1.0, raising=False)

    def failing_process_pdf(*args, **kwargs):
        return MineruProcessResult(
            success=False,
            output_dir=kwargs["output_dir"],
            output_files={},
            stdout="",
            stderr="error",
            error="boom",
        )

    monkeypatch.setattr(validation, "process_pdf", failing_process_pdf, raising=False)
    monkeypatch.setattr(validation.time, "perf_counter", lambda: 0.0, raising=False)

    with pytest.raises(ValidationFailure):
        validation.validate_pdf_processing(pdf_file, output_root=tmp_path)


def test_metrics_collector_generates_report(tmp_path: Path) -> None:
    collector = MetricsCollector(output_dir=tmp_path, sample_interval=0.01)
    collector.start()
    collector.record_pdf_start("doc.pdf", "worker-1", 1)
    time.sleep(0.02)
    collector.record_pdf_end("doc.pdf", "worker-1", True, attempts=1)
    collector.stop()

    summary = SimpleNamespace(
        total=1,
        processed=1,
        succeeded=1,
        failed=0,
        skipped=0,
        duration_seconds=1.0,
    )
    report_path = collector.generate_report(summary=summary, profile="balanced")
    assert report_path.exists()
