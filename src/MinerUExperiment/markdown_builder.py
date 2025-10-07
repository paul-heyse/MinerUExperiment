"""Utilities for constructing structured Markdown from MinerU content blocks."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, MutableSequence, Sequence

LOGGER = logging.getLogger(__name__)


class MarkdownGenerationError(RuntimeError):
    """Raised when structured Markdown cannot be generated."""


Block = Mapping[str, Any]


def load_content_list(path: Path | str) -> List[Block]:
    """Load and validate a MinerU ``content_list.json`` file.

    Parameters
    ----------
    path:
        Path to the JSON document.

    Returns
    -------
    List[Block]
        Parsed list of content blocks.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    MarkdownGenerationError
        If the JSON is invalid or does not describe a list of blocks.
    """

    content_path = Path(path)
    if not content_path.exists():
        raise FileNotFoundError(f"content_list.json not found: {content_path}")

    try:
        with content_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise MarkdownGenerationError(
            f"Invalid JSON in {content_path}: {exc.msg}"
        ) from exc

    if not isinstance(data, list):
        raise MarkdownGenerationError(
            f"Expected list of blocks in {content_path}, got {type(data).__name__}"
        )

    return data


def blocks_to_markdown(blocks: Sequence[Block]) -> str:
    """Transform a sequence of MinerU blocks into Markdown text."""

    rendered_blocks: MutableSequence[str] = []

    for block in blocks:
        if not isinstance(block, Mapping):  # pragma: no cover - defensive
            LOGGER.warning("Skipping non-mapping block: %r", block)
            continue

        block_type = str(block.get("type") or "text").lower()

        handler = _BLOCK_RENDERERS.get(block_type, _render_unknown)
        rendered = handler(block)

        if rendered:
            rendered_blocks.append(rendered)

    markdown = "\n\n".join(rendered_blocks)
    return _normalize_markdown(markdown)


def generate_structured_markdown(
    content_list_path: Path | str,
    *,
    output_path: Path | None = None,
) -> Path:
    """Generate a ``.structured.md`` file from *content_list_path*.

    Parameters
    ----------
    content_list_path:
        Path to the ``*_content_list.json`` file.
    output_path:
        Optional explicit output path. If omitted, the output path is derived
        from *content_list_path*.
    """

    content_path = Path(content_list_path)
    blocks = load_content_list(content_path)
    markdown = blocks_to_markdown(blocks)

    if output_path is None:
        output_path = structured_markdown_path(content_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    LOGGER.info(
        "Generated structured Markdown: %s -> %s", content_path, output_path
    )

    return output_path


def structured_markdown_path(content_list_path: Path | str) -> Path:
    """Return the derived ``.structured.md`` path for *content_list_path*."""

    content_path = Path(content_list_path)
    name = content_path.name
    if name.endswith("_content_list.json"):
        base = name[: -len("_content_list.json")]
        new_name = f"{base}.structured.md"
    else:
        new_name = f"{content_path.stem}.structured.md"
    return content_path.with_name(new_name)


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Iterable):
        return "\n".join(str(part).strip() for part in value if part)
    return str(value).strip()


def _render_text(block: Block) -> str:
    text_level = block.get("text_level")
    content = _extract_text(block.get("content"))
    if not content:
        return ""

    try:
        level = int(text_level)
    except (TypeError, ValueError):
        level = 0

    if level <= 0:
        return content

    level = max(1, min(level, 6))
    heading_prefix = "#" * level
    return f"{heading_prefix} {content}"


def _render_table(block: Block) -> str:
    html = _extract_text(block.get("html"))
    if not html:
        html = _extract_text(block.get("content"))

    if not html:
        LOGGER.warning("Skipping empty table block")
        return ""

    return html


def _render_equation(block: Block) -> str:
    latex = _extract_text(block.get("latex")) or _extract_text(block.get("content"))
    if not latex:
        LOGGER.warning("Skipping empty equation block")
        return ""

    stripped = latex.strip()
    if stripped.startswith("$$") and stripped.endswith("$$"):
        body = stripped[2:-2].strip()
        return f"$$\n{body}\n$$"

    return f"$$\n{stripped}\n$$"


def _render_image(block: Block) -> str:
    path = _extract_text(block.get("img_path") or block.get("path"))
    if not path:
        LOGGER.warning("Skipping image block without img_path")
        return ""

    caption_raw = block.get("img_caption") or block.get("caption")
    if isinstance(caption_raw, (list, tuple)):
        caption_parts = [
            str(part).strip() for part in caption_raw if str(part).strip()
        ]
        caption = " ".join(caption_parts)
    else:
        caption = _extract_text(caption_raw)

    alt_text = caption or ""
    parts = [f"![{alt_text}]({path})"]
    if caption:
        parts.append(f"*{caption}*")
    return "\n".join(parts)


def _render_image_caption(block: Block) -> str:
    caption_raw = block.get("content") or block.get("img_caption")
    if isinstance(caption_raw, (list, tuple)):
        caption = " ".join(str(part).strip() for part in caption_raw if str(part).strip())
    else:
        caption = _extract_text(caption_raw)
    if not caption:
        return ""
    return f"*{caption}*"


def _render_code(block: Block) -> str:
    content = _extract_text(block.get("content"))
    if not content:
        return ""

    language = _extract_text(block.get("language") or block.get("lang"))
    fence = f"```{language}\n" if language else "```\n"
    return f"{fence}{content}\n```"


def _render_list(block: Block) -> str:
    content = block.get("content")
    if isinstance(content, list):
        items = []
        for item in content:
            text = _extract_text(item)
            if text:
                items.append(f"- {text}")
        return "\n".join(items)
    text = _extract_text(content)
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(f"- {line}" for line in lines)


def _render_unknown(block: Block) -> str:
    block_type = block.get("type")
    LOGGER.warning("Encountered unknown block type: %s", block_type)
    return _extract_text(block.get("content"))


def _normalize_markdown(markdown: str) -> str:
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text.endswith("\n"):
        text += "\n"
    return text


_BLOCK_RENDERERS: dict[str, Callable[[Block], str]] = {
    "text": _render_text,
    "paragraph": _render_text,
    "table": _render_table,
    "equation": _render_equation,
    "image": _render_image,
    "image_caption": _render_image_caption,
    "code": _render_code,
    "algorithm": _render_code,
    "list": _render_list,
}

