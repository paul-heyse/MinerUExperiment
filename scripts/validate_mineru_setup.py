#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from MinerUExperiment import ValidationFailure, validate_pdf_processing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate MinerU + vLLM integration using a sample PDF.",
    )
    parser.add_argument(
        "pdf",
        type=Path,
        help="Path to the PDF that should be processed during validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where validation outputs (cold/warm) should be written.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress human-readable summary and only emit JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        report = validate_pdf_processing(args.pdf, output_root=args.output_dir)
    except FileNotFoundError as exc:
        logging.error("File not found: %s", exc)
        return 2
    except ValidationFailure as exc:
        logging.error("Validation failed: %s", exc)
        return 1

    if not args.no_summary:
        baseline = report.baseline.duration_seconds
        warmed = report.warmed.duration_seconds
        logging.info(
            "Baseline run: %.2fs | Warmed run: %.2fs | Warmup: %.2fs",
            baseline,
            warmed,
            report.warmup_duration_seconds,
        )
        if report.speedup_ratio() is not None:
            logging.info(
                "Warm run speedup: %.2fx (warmed faster: %s)",
                report.speedup_ratio(),
                "yes" if report.warmed_is_faster() else "no",
            )
        if report.gpu_snapshot_before:
            logging.info("GPU before:\n%s", report.gpu_snapshot_before)
        if report.gpu_snapshot_after:
            logging.info("GPU after:\n%s", report.gpu_snapshot_after)

    print(report.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
