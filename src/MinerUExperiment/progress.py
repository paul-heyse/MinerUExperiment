"""Utilities for consistent tqdm progress reporting throughout the project."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Optional

from tqdm.auto import tqdm


class ProgressBar:
    """Light wrapper around :class:`tqdm.tqdm` with graceful disable support."""

    def __init__(
        self,
        *,
        total: Optional[int] = None,
        enabled: bool = True,
        leave: bool = True,
        position: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._bar = (
            tqdm(total=total, leave=leave, position=position, **kwargs)
            if enabled
            else None
        )

    def update(self, value: int = 1) -> None:
        if self._bar is not None:
            self._bar.update(value)

    def set_description(self, description: str, refresh: bool = True) -> None:
        if self._bar is not None:
            self._bar.set_description(description, refresh=refresh)

    def set_postfix(self, data: Optional[Dict[str, Any]] = None, refresh: bool = True) -> None:
        if self._bar is not None and data is not None:
            self._bar.set_postfix(data, refresh=refresh)

    def write(self, message: str) -> None:
        if self._bar is not None:
            self._bar.write(message)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[override]
        self.close()


def progress_iterable(
    iterable: Iterable[Any],
    *,
    enabled: bool = True,
    **kwargs: Any,
) -> Iterator[Any]:
    """Wrap *iterable* with :func:`tqdm.auto.tqdm` when *enabled* is True."""

    if not enabled:
        yield from iterable
        return

    for item in tqdm(iterable, **kwargs):
        yield item


__all__ = [
    "ProgressBar",
    "progress_iterable",
]

