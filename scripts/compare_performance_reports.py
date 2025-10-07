#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (SRC_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from MinerUExperiment.metrics import compare_performance_reports


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple MinerU performance reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        help="Paths to performance_report.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination for aggregated comparison report",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        destination = compare_performance_reports(args.reports, output_path=args.output)
    except Exception as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 1
    print(f"Comparison report written to {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
