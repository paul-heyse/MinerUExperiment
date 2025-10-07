import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import MinerUExperiment.batch_processor as batch_processor
from MinerUExperiment.batch_processor import BatchProcessor, BatchProcessorConfig, BatchSummary
from MinerUExperiment.worker_coordinator import (
    done_path_for,
    failed_path_for,
    lock_path_for,
)


STUB_SCRIPT = """#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path


def _load_behavior(pdf_path: Path) -> dict:
    behavior_path = pdf_path.with_suffix(pdf_path.suffix + ".behavior.json")
    if behavior_path.exists():
        try:
            return json.loads(behavior_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pdf", dest="pdf_path", required=True)
    parser.add_argument("-o", "--output", dest="output_dir", required=True)
    parser.add_argument("-b", "--backend", dest="backend", required=True)
    args, _ = parser.parse_known_args()

    pdf_path = Path(args.pdf_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    behavior = _load_behavior(pdf_path)
    sleep = float(behavior.get("sleep", os.environ.get("MINERU_TEST_SLEEP", 0)))
    failures_before_success = int(behavior.get("failures_before_success", 0))
    permanent_failure = bool(behavior.get("permanent_failure", False))

    attempts_path = pdf_path.with_suffix(pdf_path.suffix + ".attempts")
    attempts = 0
    if attempts_path.exists():
        try:
            attempts = int(attempts_path.read_text(encoding="utf-8"))
        except ValueError:
            attempts = 0
    attempts += 1
    attempts_path.write_text(str(attempts), encoding="utf-8")

    if sleep:
        time.sleep(float(sleep))

    if permanent_failure:
        sys.stderr.write(f"permanent failure for {pdf_path.name} on attempt {attempts}\\n")
        return 1

    if attempts <= failures_before_success:
        sys.stderr.write(
            f"transient failure for {pdf_path.name} on attempt {attempts}/{failures_before_success}\\n"
        )
        return 1

    stem = pdf_path.stem
    (output_dir / f"{stem}.md").write_text(f"# {stem}\\nProcessed", encoding="utf-8")
    content_list = [
        {"type": "text", "content": f"{stem} Title", "text_level": 1},
        {"type": "text", "content": "Body paragraph."},
    ]
    (output_dir / f"{stem}_content_list.json").write_text(
        json.dumps(content_list), encoding="utf-8"
    )
    (output_dir / f"{stem}_middle.json").write_text("{}", encoding="utf-8")
    (output_dir / f"{stem}_model.json").write_text("{}", encoding="utf-8")
    (output_dir / f"{stem}_layout.pdf").write_bytes(b"%PDF-1.4\\n%")
    (output_dir / f"{stem}_origin.pdf").write_bytes(b"%PDF-1.4\\n%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""


class DummyMetrics:
    def __init__(self, *, output_dir: Path, sample_interval: float, gpu_index: int, benchmark_mode: bool):
        self.output_dir = output_dir
        self.sample_interval = sample_interval
        self.gpu_index = gpu_index
        self.benchmark_mode = benchmark_mode
        self.started = False

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started = True

    def stop(self) -> None:
        self.started = False

    def record_pdf_start(self, *_, **__) -> None:  # pragma: no cover - noop
        return

    def record_pdf_end(self, *_, **__) -> None:  # pragma: no cover - noop
        return

    def generate_report(self, *, summary: BatchSummary, profile: str) -> Path:
        report_path = self.output_dir / "report.json"
        data = {
            "profile": profile,
            "processed": summary.processed,
            "succeeded": summary.succeeded,
            "failed": summary.failed,
        }
        report_path.write_text(json.dumps(data), encoding="utf-8")
        return report_path


class DummyMineruConfig:
    def __init__(self) -> None:
        self.vllm_settings = type(
            "VllmSettings",
            (),
            {
                "data_parallel_size": 1,
                "tensor_parallel_size": 1,
                "max_model_len": 16384,
                "dtype": "float16",
                "gpu_memory_utilization": 0.9,
                "block_size": 16,
                "swap_space_mb": 8192,
            },
        )()

    def update_vllm_settings(self, **settings: object) -> None:
        for key, value in settings.items():
            setattr(self.vllm_settings, key, value)


@pytest.fixture(autouse=True)
def stub_batch_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch_processor, "MetricsCollector", DummyMetrics, raising=False)
    monkeypatch.setattr(batch_processor, "load_config", lambda: DummyMineruConfig(), raising=False)
    monkeypatch.setattr(batch_processor, "write_config", lambda config: None, raising=False)
    monkeypatch.setattr(
        BatchProcessor,
        "_resolve_dtype_preference",
        lambda self: "float16",
        raising=False,
    )
    monkeypatch.setattr(
        BatchProcessor,
        "_start_resource_monitor",
        lambda self: None,
        raising=False,
    )
    monkeypatch.setattr(
        BatchProcessor,
        "_stop_resource_monitor",
        lambda self: None,
        raising=False,
    )


@pytest.fixture
def fake_mineru(tmp_path: Path) -> Path:
    script_path = tmp_path / "fake_mineru.py"
    script_path.write_text(STUB_SCRIPT, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    return script_path


def _create_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n%EOF\n")


def _set_behavior(path: Path, *, sleep: float | None = None, failures_before_success: int = 0, permanent_failure: bool = False) -> None:
    behavior = {
        "failures_before_success": failures_before_success,
        "permanent_failure": permanent_failure,
    }
    if sleep is not None:
        behavior["sleep"] = sleep
    path.with_suffix(path.suffix + ".behavior.json").write_text(json.dumps(behavior), encoding="utf-8")


def _attempts_for(path: Path) -> int:
    attempts_path = path.with_suffix(path.suffix + ".attempts")
    if not attempts_path.exists():
        return 0
    return int(attempts_path.read_text(encoding="utf-8"))


def test_batch_processor_processes_pdfs_without_duplicates(tmp_path: Path, fake_mineru: Path) -> None:
    input_dir = tmp_path / "PDFsToProcess"
    output_dir = tmp_path / "MDFilesCreated"
    input_dir.mkdir()

    success_pdfs = []
    for index in range(5):
        pdf = input_dir / f"doc_{index}.pdf"
        _create_pdf(pdf)
        success_pdfs.append(pdf)

    nested_pdf = input_dir / "nested" / "deep_doc.pdf"
    _create_pdf(nested_pdf)
    success_pdfs.append(nested_pdf)

    retry_pdf = input_dir / "retry_me.pdf"
    _create_pdf(retry_pdf)
    _set_behavior(retry_pdf, sleep=0.05, failures_before_success=2)
    success_pdfs.append(retry_pdf)

    failure_pdf = input_dir / "always_fail.pdf"
    _create_pdf(failure_pdf)
    _set_behavior(failure_pdf, sleep=0.05, permanent_failure=True)

    config = BatchProcessorConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        workers=2,
        poll_interval=0.1,
        max_retries=3,
        retry_delay=0.05,
        progress_interval=0.2,
        mineru_cli=str(fake_mineru),
        mineru_backend="test-backend",
        mineru_extra_args=tuple(),
        env_overrides={},
        log_progress=False,
        memory_pause_threshold=0.95,
        memory_resume_threshold=0.5,
        cpu_pause_threshold=0.99,
        resource_monitor_interval=0.2,
        worker_memory_limit_gb=0.1,
        dtype="float16",
    )

    processor = BatchProcessor(config)
    summary = processor.run()

    total_pdfs = len(success_pdfs) + 1
    assert summary.total == total_pdfs
    assert summary.processed == total_pdfs
    assert summary.succeeded == len(success_pdfs)
    assert summary.failed == 1

    for pdf in success_pdfs:
        assert done_path_for(pdf).exists()
        relative = pdf.relative_to(input_dir)
        document_dir = output_dir / relative.parent
        stem = relative.stem
        assert (document_dir / f"{stem}.md").exists()
        assert (document_dir / "content_list.json").exists()
        assert (document_dir / "middle.json").exists()
        assert (document_dir / "model.json").exists()
        structured_path = document_dir / f"{stem}.structured.md"
        assert structured_path.exists()
        structured_contents = structured_path.read_text(encoding="utf-8")
        assert f"# {stem} Title" in structured_contents
        assert _attempts_for(pdf) == (3 if pdf == retry_pdf else 1)

    assert failed_path_for(failure_pdf).exists()
    assert _attempts_for(failure_pdf) == 4

    assert not any(input_dir.rglob(f"*{lock_path_for(success_pdfs[0]).suffix}"))


def test_batch_processor_graceful_shutdown(tmp_path: Path, fake_mineru: Path) -> None:
    input_dir = tmp_path / "PDFsToProcess"
    output_dir = tmp_path / "MDFilesCreated"
    input_dir.mkdir()

    pdfs = []
    for index in range(6):
        pdf = input_dir / f"slow_{index}.pdf"
        _create_pdf(pdf)
        _set_behavior(pdf, sleep=0.4)
        pdfs.append(pdf)

    config = BatchProcessorConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        workers=2,
        poll_interval=0.1,
        max_retries=1,
        retry_delay=0.05,
        progress_interval=0.2,
        mineru_cli=str(fake_mineru),
        mineru_backend="test-backend",
        mineru_extra_args=tuple(),
        env_overrides={},
        log_progress=False,
        memory_pause_threshold=0.95,
        memory_resume_threshold=0.5,
        cpu_pause_threshold=0.99,
        resource_monitor_interval=0.2,
        worker_memory_limit_gb=0.1,
        dtype="float16",
    )

    processor = BatchProcessor(config)

    def trigger_shutdown() -> None:
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if any(done_path_for(pdf).exists() for pdf in pdfs):
                break
            if any(_attempts_for(pdf) > 0 for pdf in pdfs):
                # Workers started but no completions yet; wait a little longer.
                time.sleep(0.05)
                continue
            time.sleep(0.05)
        time.sleep(0.1)
        processor._shutdown_event.set()

    shutdown_thread = threading.Thread(target=trigger_shutdown)
    shutdown_thread.start()
    summary = processor.run()
    shutdown_thread.join()

    assert summary.total == len(pdfs)
    assert summary.processed < len(pdfs)
    assert summary.processed == summary.succeeded
    assert summary.failed == 0

    processed_pdfs = [pdf for pdf in pdfs if done_path_for(pdf).exists()]
    assert 0 < len(processed_pdfs) < len(pdfs)

    for pdf in processed_pdfs:
        assert _attempts_for(pdf) == 1

    pending_pdfs = [pdf for pdf in pdfs if not done_path_for(pdf).exists()]
    for pdf in pending_pdfs:
        assert _attempts_for(pdf) == 0

    assert not any(input_dir.rglob("*.lock"))
