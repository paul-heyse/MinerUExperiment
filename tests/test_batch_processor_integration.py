import importlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import MinerUExperiment.batch_processor as batch_processor
from MinerUExperiment.batch_processor import (
    BatchProcessor,
    BatchProcessorConfig,
    BatchProcessorError,
)
from MinerUExperiment.worker_coordinator import (
    done_path_for,
    failed_path_for,
    lock_path_for,
)


def _create_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n%EOF\n")


def _set_behavior(
    path: Path,
    *,
    sleep: float | None = None,
    failures_before_success: int = 0,
    permanent_failure: bool = False,
) -> None:
    behavior = {
        "failures_before_success": failures_before_success,
        "permanent_failure": permanent_failure,
    }
    if sleep is not None:
        behavior["sleep"] = sleep
    path.with_suffix(path.suffix + ".behavior.json").write_text(
        json.dumps(behavior), encoding="utf-8"
    )


def _attempts_for(path: Path) -> int:
    attempts_path = path.with_suffix(path.suffix + ".attempts")
    if not attempts_path.exists():
        return 0
    return int(attempts_path.read_text(encoding="utf-8"))


def test_fake_mineru_fixture_available(fake_mineru: Path) -> None:
    """Sanity check that the stub mineru CLI fixture is usable across modules."""
    assert fake_mineru.exists()
    assert os.access(fake_mineru, os.X_OK)


def test_batch_processor_processes_pdfs_without_duplicates(
    tmp_path: Path, fake_mineru: Path
) -> None:
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

    failure_pdfs = []
    for index in range(2):
        failure_pdf = input_dir / f"always_fail_{index}.pdf"
        _create_pdf(failure_pdf)
        _set_behavior(failure_pdf, sleep=0.05, permanent_failure=True)
        failure_pdfs.append(failure_pdf)

    config = BatchProcessorConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        workers=2,
        poll_interval=0.1,
        max_retries=3,
        retry_delay=0.05,
        progress_interval=0.2,
        mineru_cli=str(fake_mineru),
        mineru_backend="pipeline",
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

    total_pdfs = len(success_pdfs) + len(failure_pdfs)
    assert summary.total == total_pdfs
    assert summary.processed == total_pdfs
    assert summary.succeeded == len(success_pdfs)
    assert summary.failed == len(failure_pdfs)
    assert summary.skipped == 0
    assert summary.duration_seconds > 0.0

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

    assert len(summary.failures) == len(failure_pdfs)
    for failure_pdf in failure_pdfs:
        assert failed_path_for(failure_pdf).exists()
        assert _attempts_for(failure_pdf) == config.max_retries + 1

    failure_report_path = output_dir / "failed_documents.json"
    assert summary.failure_report == failure_report_path
    assert failure_report_path.exists()
    report_data = json.loads(failure_report_path.read_text(encoding="utf-8"))
    assert report_data["total_failures"] == len(failure_pdfs)
    reported_pdfs = {entry["pdf"] for entry in report_data["failures"]}
    expected_pdfs = {str(pdf.relative_to(input_dir)) for pdf in failure_pdfs}
    assert reported_pdfs == expected_pdfs
    assert {entry["pdf"] for entry in summary.failures} == expected_pdfs
    for entry in report_data["failures"]:
        assert entry["attempt_count"] == config.max_retries + 1
        assert entry["final_error"]
        assert entry["attempts"]
        assert entry["attempts"][-1]["number"] == config.max_retries + 1

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
        mineru_backend="pipeline",
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
    assert summary.failures == []
    assert summary.failure_report is None
    assert summary.skipped == 0
    assert summary.duration_seconds > 0.0
    assert not (output_dir / "failed_documents.json").exists()

    processed_pdfs = [pdf for pdf in pdfs if done_path_for(pdf).exists()]
    assert 0 < len(processed_pdfs) < len(pdfs)

    for pdf in processed_pdfs:
        assert _attempts_for(pdf) == 1

    pending_pdfs = [pdf for pdf in pdfs if not done_path_for(pdf).exists()]
    for pdf in pending_pdfs:
        assert _attempts_for(pdf) == 0


def _restore_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        BatchProcessor,
        "_run_preflight_checks",
        batch_processor._ORIGINAL_PREFLIGHT,
        raising=False,
    )


def _stub_dependency_imports(
    monkeypatch: pytest.MonkeyPatch, *, missing: set[str] | None = None
) -> None:
    missing = missing or set()
    modules_to_manage = {"torch"} | set(missing)
    for name in modules_to_manage:
        if name in missing:
            monkeypatch.delitem(sys.modules, name, raising=False)

    original_import = importlib.import_module
    original_find_spec = importlib.util.find_spec
    module_cache: dict[str, ModuleType] = {}

    def _fake_import(name: str, package: str | None = None):
        if name in modules_to_manage:
            if name in missing:
                raise ModuleNotFoundError(name)
            if name not in module_cache:
                module = ModuleType(name)
                if name == "torch":
                    module.cuda = type(  # type: ignore[attr-defined]
                        "_Cuda", (), {"is_available": staticmethod(lambda: False), "device_count": staticmethod(lambda: 0)}
                    )()
                module_cache[name] = module
                monkeypatch.setitem(sys.modules, name, module)
            return module_cache[name]
        return original_import(name, package)  # type: ignore[call-arg]

    def _fake_find_spec(name: str, package: str | None = None):
        if name in missing:
            return None
        if name in modules_to_manage:
            return SimpleNamespace(name=name)  # type: ignore[return-value]
        return original_find_spec(name, package)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)


def test_batch_processor_missing_cli_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "PDFsToProcess"
    output_dir = tmp_path / "MDFilesCreated"
    input_dir.mkdir()
    _create_pdf(input_dir / "doc.pdf")

    _restore_preflight(monkeypatch)
    _stub_dependency_imports(monkeypatch)

    config = BatchProcessorConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        workers=1,
        poll_interval=0.1,
        max_retries=0,
        retry_delay=0.05,
        progress_interval=0.2,
        mineru_cli="/not/a/real/mineru",
        mineru_backend="pipeline",
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

    with pytest.raises(BatchProcessorError, match="MinerU CLI executable not found"):
        processor.run()

    assert processor._workers == []


def test_batch_processor_missing_dependency_aborts(
    tmp_path: Path, fake_mineru: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "PDFsToProcess"
    output_dir = tmp_path / "MDFilesCreated"
    input_dir.mkdir()
    _create_pdf(input_dir / "doc.pdf")

    _restore_preflight(monkeypatch)
    _stub_dependency_imports(monkeypatch, missing={"torch"})

    config = BatchProcessorConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        workers=1,
        poll_interval=0.1,
        max_retries=0,
        retry_delay=0.05,
        progress_interval=0.2,
        mineru_cli=str(fake_mineru),
        mineru_backend="pipeline",
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

    with pytest.raises(BatchProcessorError, match="Missing required Python modules"):
        processor.run()

    assert processor._workers == []

    assert not any(input_dir.rglob("*.lock"))
