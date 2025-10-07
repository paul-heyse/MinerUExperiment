import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from MinerUExperiment.batch_processor import BatchSummary
from MinerUExperiment.worker_coordinator import done_path_for
from scripts import process_batch


def test_parse_args_defaults() -> None:
    args = process_batch.parse_args([])
    assert args.input_dir == Path("PDFsToProcess")
    assert args.output_dir == Path("MDFilesCreated")
    assert args.workers is None
    assert args.max_retries == 3
    assert args.retry_delay == pytest.approx(10.0)
    assert args.extra_arg is None
    assert args.env is None
    assert args.profile == "balanced"
    assert not args.no_progress


def test_build_config_applies_profile_and_overrides(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()

    args = process_batch.parse_args(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--workers",
            "3",
            "--max-retries",
            "5",
            "--retry-delay",
            "2.5",
            "--poll-interval",
            "0.4",
            "--progress-interval",
            "0.7",
            "--backend",
            "custom-backend",
            "--mineru-cli",
            "mineru-cli",
            "--extra-arg=--alpha",
            "--extra-arg=--beta",
            "--env",
            "OMP_NUM_THREADS=32",
            "--env",
            "CUSTOM=VALUE",
            "--profile",
            "throughput",
            "--benchmark",
        ]
    )

    config = process_batch.build_config(args)

    assert config.input_dir == input_dir.resolve()
    assert config.output_dir == output_dir.resolve()
    assert config.workers == 3
    assert config.max_retries == 5
    assert config.retry_delay == pytest.approx(2.5)
    assert config.poll_interval == pytest.approx(0.4)
    assert config.progress_interval == pytest.approx(0.7)
    assert config.mineru_backend == "custom-backend"
    assert config.mineru_cli == "mineru-cli"
    assert config.profile == "throughput"
    assert config.benchmark is True
    assert config.performance_report_path == output_dir / "performance_report.json"

    assert config.mineru_extra_args[:2] == ("--alpha", "--beta")
    assert config.mineru_extra_args[2:] == ("--batch-mode", "aggressive")

    assert config.env_overrides["CUSTOM"] == "VALUE"
    assert config.env_overrides["OMP_NUM_THREADS"] == "32"
    assert "PYTORCH_CUDA_ALLOC_CONF" in config.env_overrides


def test_main_returns_failure_code_on_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()

    class FakeBatchProcessor:
        def __init__(self, config: process_batch.BatchProcessorConfig) -> None:
            self.config = config

        def run(self) -> BatchSummary:
            return BatchSummary(total=1, processed=1, succeeded=0, failed=1, skipped=0, duration_seconds=1.0)

    monkeypatch.setattr(process_batch, "BatchProcessor", FakeBatchProcessor, raising=False)

    exit_code = process_batch.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 2


def test_main_profile_info(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = process_batch.main(["--profile-info"])
    out, err = capsys.readouterr()
    assert exit_code == 0
    assert "Performance profiles:" in out
    assert err == ""


def test_main_end_to_end_success(tmp_path: Path, fake_mineru: Path) -> None:
    input_dir = tmp_path / "PDFsToProcess"
    output_dir = tmp_path / "MDFilesCreated"
    input_dir.mkdir()

    pdf_one = input_dir / "doc1.pdf"
    pdf_one.write_bytes(b"%PDF-1.4\n%EOF")
    pdf_two = input_dir / "nested" / "doc2.pdf"
    pdf_two.parent.mkdir(parents=True, exist_ok=True)
    pdf_two.write_bytes(b"%PDF-1.4\n%EOF")

    exit_code = process_batch.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--workers",
            "1",
            "--mineru-cli",
            str(fake_mineru),
            "--backend",
            "test-backend",
            "--poll-interval",
            "0.1",
            "--progress-interval",
            "0.2",
            "--retry-delay",
            "0.05",
        ]
    )

    assert exit_code == 0

    for pdf in (pdf_one, pdf_two):
        assert done_path_for(pdf).exists()
        relative = pdf.relative_to(input_dir)
        document_dir = output_dir / relative.parent
        stem = relative.stem
        structured_path = document_dir / f"{stem}.structured.md"
        assert structured_path.exists()
        contents = structured_path.read_text(encoding="utf-8")
        assert f"# {stem} Title" in contents

    report_path = output_dir / "performance_report.json"
    assert report_path.exists()

    lock_files = list(input_dir.rglob("*.lock"))
    assert not lock_files
