from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import MinerUExperiment.batch_processor as batch_processor
from MinerUExperiment.batch_processor import BatchProcessor, BatchSummary


ORIGINAL_PREFLIGHT = BatchProcessor._run_preflight_checks


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
    omit_outputs = bool(behavior.get("omit_outputs", False))

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
        sys.stderr.write(
            f"permanent failure for {pdf_path.name} on attempt {attempts}\\n"
        )
        return 1

    if attempts <= failures_before_success:
        sys.stderr.write(
            f"transient failure for {pdf_path.name} on attempt {attempts}/{failures_before_success}\\n"
        )
        return 1

    if omit_outputs:
        return 0

    stem = pdf_path.stem
    (output_dir / f"{stem}.md").write_text(
        f"# {stem}\\nProcessed", encoding="utf-8"
    )
    content_list = [
        {"type": "text", "content": f"{stem} Title", "text_level": 1},
        {"type": "text", "content": "Body paragraph."},
    ]
    (output_dir / f"{stem}_content_list.json").write_text(
        json.dumps(content_list), encoding="utf-8"
    )
    (output_dir / f"{stem}_middle.json").write_text("{}", encoding="utf-8")
    (output_dir / f"{stem}_model.json").write_text("{}", encoding="utf-8")
    (output_dir / f"{stem}_layout.pdf").write_bytes(b"%PDF-1.4%")
    (output_dir / f"{stem}_origin.pdf").write_bytes(b"%PDF-1.4%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""


class DummyMetrics:
    def __init__(self, *, output_dir: Path, sample_interval: float, benchmark_mode: bool):
        self.output_dir = output_dir
        self.sample_interval = sample_interval
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
    monkeypatch.setattr(batch_processor, "warmup_gpu", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        batch_processor,
        "_ORIGINAL_PREFLIGHT",
        ORIGINAL_PREFLIGHT,
        raising=False,
    )
    monkeypatch.setattr(
        BatchProcessor,
        "_resolve_dtype_preference",
        lambda self: "float16",
        raising=False,
    )
    monkeypatch.setattr(
        BatchProcessor,
        "_run_preflight_checks",
        lambda self: {"preflight": "skipped"},
        raising=False,
    )
    monkeypatch.setattr(BatchProcessor, "_validate_resources", lambda self: None, raising=False)
    monkeypatch.setattr(BatchProcessor, "_start_resource_monitor", lambda self: None, raising=False)
    monkeypatch.setattr(BatchProcessor, "_stop_resource_monitor", lambda self: None, raising=False)


@pytest.fixture
def fake_mineru(tmp_path: Path) -> Path:
    script_path = tmp_path / "fake_mineru.py"
    script_path.write_text(STUB_SCRIPT, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    return script_path
