"""Utilities for consistent tqdm progress reporting throughout the project."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Optional

try:  # pragma: no cover - exercised indirectly via import path
    from tqdm.auto import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    class _TqdmShim:
        """Fallback no-op progress helper when tqdm is unavailable."""

        def __init__(self, iterable=None, **_: Any) -> None:
            self._iterable = iterable

        def __iter__(self) -> Iterator[Any]:
            if self._iterable is None:
                return iter(())
            return iter(self._iterable)

        def update(self, value: int = 1) -> None:  # noqa: ARG002 - keep signature parity
            return None

        def set_description(self, description: str, *, refresh: bool = True) -> None:
            return None

        def set_postfix(self, data: Optional[Dict[str, Any]] = None, *, refresh: bool = True) -> None:
            return None

        def write(self, message: str) -> None:
            return None

        def close(self) -> None:
            return None

        def __enter__(self) -> "_TqdmShim":
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[override]
            return None

    def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        return _TqdmShim(iterable, **kwargs)


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
