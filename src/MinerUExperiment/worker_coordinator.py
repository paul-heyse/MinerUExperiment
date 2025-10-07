from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

LOGGER = logging.getLogger(__name__)

LOCK_EXTENSION = ".lock"
DONE_EXTENSION = ".done"
FAILED_EXTENSION = ".failed"
PDF_SUFFIX = ".pdf"


def _is_hidden(relative_path: Path) -> bool:
    """Return True when the relative path (or any parent) is hidden."""

    return any(part.startswith(".") for part in relative_path.parts if part)


def _atomic_write_file(path: Path, content: str) -> None:
    """Atomically create a file with *content*.

    The file is created with O_EXCL to ensure only one process succeeds.
    """

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    # Use 0o644 permissions so other workers can read stale lock metadata.
    fd = os.open(path, flags, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
    except Exception:
        # Only attempt to close the descriptor on failure; fdopen handles on success.
        os.close(fd)
        raise


@dataclass(frozen=True)
class CoordinatorConfig:
    input_dir: Path
    stale_after_seconds: int = 30 * 60

    def __post_init__(self) -> None:
        input_dir = self.input_dir.expanduser().resolve()
        object.__setattr__(self, "input_dir", input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")


def discover_pdfs(root: Path) -> List[Path]:
    """Return all PDF files (case-insensitive) beneath *root*.

    Hidden files/directories are ignored. Results are sorted for determinism.
    """

    pdfs: List[Path] = []
    for current_dir, dirnames, filenames in os.walk(root):
        current_path = Path(current_dir)
        relative = current_path.relative_to(root)
        if _is_hidden(relative):
            # Prevent os.walk from traversing hidden directories by clearing dirnames.
            dirnames[:] = []
            continue

        # Remove hidden subdirectories in-place to block traversal.
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]

        for filename in filenames:
            if filename.startswith("."):
                continue
            if not filename.lower().endswith(PDF_SUFFIX):
                continue
            pdfs.append(current_path / filename)

    pdfs.sort()
    return pdfs


def lock_path_for(pdf_path: Path) -> Path:
    return pdf_path.with_suffix(pdf_path.suffix + LOCK_EXTENSION)


def done_path_for(pdf_path: Path) -> Path:
    return pdf_path.with_suffix(pdf_path.suffix + DONE_EXTENSION)


def failed_path_for(pdf_path: Path) -> Path:
    return pdf_path.with_suffix(pdf_path.suffix + FAILED_EXTENSION)


class WorkerCoordinator:
    """Coordinate which worker processes which PDF via file locks."""

    def __init__(self, config: CoordinatorConfig):
        self.config = config

    # -- Discovery -----------------------------------------------------------------

    def pending_items(self) -> List[Path]:
        """Return all PDFs that are not currently locked or completed."""

        pending: List[Path] = []
        for pdf in discover_pdfs(self.config.input_dir):
            if lock_path_for(pdf).exists():
                continue
            if done_path_for(pdf).exists():
                continue
            if failed_path_for(pdf).exists():
                continue
            pending.append(pdf)
        return pending

    # -- Claim/release --------------------------------------------------------------

    def claim(self, pdf_path: Path, worker_id: str) -> bool:
        """Attempt to claim *pdf_path* for *worker_id*.

        Returns True on success, False if another worker holds the lock.
        """

        lock_path = lock_path_for(pdf_path)
        timestamp = time.time()
        metadata = f"worker_id={worker_id}\ntimestamp={timestamp}\n"

        try:
            _atomic_write_file(lock_path, metadata)
        except FileExistsError:
            LOGGER.debug("Worker %s failed to claim %s (lock exists)", worker_id, pdf_path)
            return False
        except FileNotFoundError:
            # Parent directories may have been removed between discovery and claim.
            LOGGER.warning("Input file disappeared before claim: %s", pdf_path)
            return False

        LOGGER.info("Worker %s claimed %s", worker_id, pdf_path)
        return True

    def release(
        self,
        pdf_path: Path,
        *,
        worker_id: str,
        success: bool,
        permanent: bool = False,
        message: str | None = None,
    ) -> None:
        """Release lock for *pdf_path* and mark completion status."""

        lock_path = lock_path_for(pdf_path)
        if lock_path.exists():
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

        if success or permanent:
            status = "success" if success else "failed"
            target_path = done_path_for(pdf_path) if success else failed_path_for(pdf_path)
            details = (
                f"worker_id={worker_id}\nstatus={status}\ncompleted_at={time.time()}\n"
            )
            if message:
                details += f"message={message}\n"
            try:
                target_path.write_text(details, encoding="utf-8")
            except Exception:
                LOGGER.exception("Failed to write completion marker for %s", pdf_path)
            else:
                LOGGER.info(
                    "Worker %s marked %s as %s", worker_id, pdf_path, status
                )

    # -- Stale lock management ------------------------------------------------------

    def clean_stale_locks(self) -> List[Path]:
        """Remove lock files older than the configured threshold.

        Returns a list of PDFs whose locks were reclaimed.
        """

        now = time.time()
        reclaimed: List[Path] = []
        for lock in self._iter_lock_files():
            try:
                mtime = lock.stat().st_mtime
            except FileNotFoundError:
                continue

            age = now - mtime
            if age < self.config.stale_after_seconds:
                continue

            pdf_path = Path(str(lock)[: -len(LOCK_EXTENSION)])
            try:
                lock.unlink()
            except FileNotFoundError:
                continue

            reclaimed.append(pdf_path)
            LOGGER.warning(
                "Reclaimed stale lock for %s (age %.0fs)", pdf_path, age
            )

        return reclaimed

    # -- Helpers -------------------------------------------------------------------

    def _iter_lock_files(self) -> Iterator[Path]:
        pattern = f"*{LOCK_EXTENSION}"
        yield from self.config.input_dir.rglob(pattern)

    # -- Public API ----------------------------------------------------------------

    def acquire_next(self, worker_id: str) -> Path | None:
        """Return the next available PDF and claim it.

        If no PDFs are available, returns None.
        """

        for pdf in self.pending_items():
            if self.claim(pdf, worker_id=worker_id):
                return pdf
        return None


def ensure_directories(directories: Sequence[Path]) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "CoordinatorConfig",
    "WorkerCoordinator",
    "discover_pdfs",
    "lock_path_for",
    "done_path_for",
    "failed_path_for",
    "ensure_directories",
]

