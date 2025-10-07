#!/usr/bin/env python3
"""Standalone script for generating structured Markdown outputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

from MinerUExperiment.markdown_builder import (
    MarkdownGenerationError,
    generate_structured_markdown,
)


LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _find_content_lists(root: Path) -> List[Path]:
    return sorted(root.rglob("*_content_list.json"))


def _process_paths(paths: Iterable[Path]) -> tuple[int, int]:
    processed = 0
    errors = 0
    for path in paths:
        LOGGER.info("Processing %s", path)
        try:
            generate_structured_markdown(path)
        except (FileNotFoundError, MarkdownGenerationError) as exc:
            LOGGER.error("Failed to process %s: %s", path, exc)
            errors += 1
        except Exception:  # pragma: no cover - unexpected
            LOGGER.exception("Unexpected error while processing %s", path)
            errors += 1
        else:
            processed += 1
    return processed, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate .structured.md files for MinerU outputs by processing "
            "*_content_list.json files recursively."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing MinerU output files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )

    args = parser.parse_args()

    _configure_logging(args.verbose)

    target_dir = args.directory.expanduser().resolve()
    if not target_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {target_dir}")

    content_lists = _find_content_lists(target_dir)
    if not content_lists:
        LOGGER.info("No *_content_list.json files found in %s", target_dir)
        return

    LOGGER.info("Found %d content list files", len(content_lists))
    processed, errors = _process_paths(content_lists)
    LOGGER.info("Structured Markdown generated for %d files", processed)
    if errors:
        LOGGER.warning("Encountered %d errors during processing", errors)
    else:
        LOGGER.info("Completed without errors")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

