from __future__ import annotations

import json
from pathlib import Path

import pytest

from MinerUExperiment.markdown_builder import (
    MarkdownGenerationError,
    blocks_to_markdown,
    generate_structured_markdown,
    load_content_list,
    structured_markdown_path,
)


def test_load_content_list_success(tmp_path: Path) -> None:
    data = [{"type": "text", "content": "Heading", "text_level": 1}]
    path = tmp_path / "doc_content_list.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    blocks = load_content_list(path)

    assert blocks == data


def test_load_content_list_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing_content_list.json"
    with pytest.raises(FileNotFoundError):
        load_content_list(path)


def test_load_content_list_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad_content_list.json"
    path.write_text("{not valid", encoding="utf-8")

    with pytest.raises(MarkdownGenerationError):
        load_content_list(path)


def test_blocks_to_markdown_handles_content_types() -> None:
    blocks = [
        {"type": "text", "content": "Title", "text_level": 1},
        {"type": "text", "content": "Overview", "text_level": 2},
        {"type": "text", "content": "Paragraph text."},
        {"type": "text", "content": "Deep Heading", "text_level": 8},
        {"type": "table", "html": "<table><tr><td>Cell</td></tr></table>"},
        {"type": "equation", "content": "E=mc^2"},
        {
            "type": "image",
            "img_path": "images/figure.png",
            "img_caption": ["Figure", "One"],
        },
        {"type": "code", "content": "print('hi')", "language": "python"},
        {"type": "list", "content": ["First", "Second"]},
        {"type": "unknown", "content": "Fallback"},
    ]

    markdown = blocks_to_markdown(blocks)

    assert "# Title" in markdown
    assert "## Overview" in markdown
    assert "Paragraph text." in markdown
    assert "###### Deep Heading" in markdown
    assert "<table><tr><td>Cell</td></tr></table>" in markdown
    assert "$$\nE=mc^2\n$$" in markdown
    assert "![Figure One](images/figure.png)" in markdown
    assert "*Figure One*" in markdown
    assert "```python\nprint('hi')\n```" in markdown
    assert "- First" in markdown and "- Second" in markdown
    assert "Fallback" in markdown

    assert "\n\n\n" not in markdown


def test_generate_structured_markdown_creates_file(tmp_path: Path) -> None:
    blocks = [
        {"type": "text", "content": "Doc Title", "text_level": 1},
        {"type": "equation", "content": "x = 1"},
    ]
    content_path = tmp_path / "doc_content_list.json"
    content_path.write_text(json.dumps(blocks), encoding="utf-8")

    output_path = generate_structured_markdown(content_path)

    expected_path = tmp_path / "doc.structured.md"
    assert output_path == expected_path
    assert expected_path.exists()

    contents = expected_path.read_text(encoding="utf-8")
    assert contents.startswith("# Doc Title")
    assert contents.strip().endswith("$$")


def test_structured_markdown_path_default(tmp_path: Path) -> None:
    path = tmp_path / "chapter_content_list.json"
    derived = structured_markdown_path(path)
    assert derived.name == "chapter.structured.md"


def test_blocks_to_markdown_tables_only() -> None:
    blocks = [{"type": "table", "content": "<table></table>"}]

    markdown = blocks_to_markdown(blocks)

    assert markdown.strip() == "<table></table>"

