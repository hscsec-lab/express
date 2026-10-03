"""Modern terminal progress helpers built on Rich."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from rich.filesize import decimal
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Column
from rich.text import Text

from express import console

if TYPE_CHECKING:
    from express.base.cleanup import CleanupScope


def short_label(text: str, max_len: int = 42) -> str:
    """Truncate long paths/tensor names for fixed-width progress rows."""
    if max_len < 2:
        return text[:max_len]
    if len(text) <= max_len:
        return text
    if max_len == 2:
        return "…" + text[-1]
    return f"…{text[-(max_len - 1):]}"


@contextmanager
def busy_status(message: str, *, spinner: str = "dots") -> Iterator:
    """Show a transient spinner status line while work runs."""
    with console.status(f"[cyan]{message}[/cyan]", spinner=spinner):
        yield


def catalog_progress() -> Progress:
    """Progress UI for scanning/loading remote model indexes."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


class TransferAmountColumn(ProgressColumn):
    """File-count for the overall row; byte sizes for active transfers."""

    def render(self, task) -> Text:
        if task.fields.get("unit") == "files":
            total = int(task.total) if task.total else 0
            done = int(task.completed)
            return Text(f"{done}/{total} files")
        total = int(task.total) if task.total else 0
        done = int(task.completed)
        if total:
            return Text(f"{decimal(done)}/{decimal(total)}")
        return Text(decimal(done))


def index_progress() -> Progress:
    """Progress UI for local folder indexing (incl. per-tensor safetensors hashing)."""
    return Progress(
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description}",
            table_column=Column(max_width=34, overflow="ellipsis"),
        ),
        TextColumn(
            "[dim]{task.fields[tensor]}[/dim]",
            table_column=Column(max_width=46, overflow="ellipsis"),
        ),
        BarColumn(bar_width=18),
        MofNCompleteColumn(),
        console=console,
        transient=True,
        expand=False,
    )


class TransferSession:
    """
    Single live progress view for multi-file push/pull.

    Shows overall file completion plus the active file's byte progress.
    """

    def __init__(self, title: str, total_files: int, *, cleanup: Optional["CleanupScope"] = None):
        self.title = title
        self.total_files = max(total_files, 0)
        self.cleanup = cleanup
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=28),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TransferAmountColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )
        self._overall_id = None
        self._current_id = None
        self._files_done = 0
        self._blobs_done = 0
        self._blobs_total = 0
        self._aggregate_label = ""
        self._lock = threading.Lock()

    def __enter__(self) -> "TransferSession":
        self.progress.start()
        self._overall_id = self.progress.add_task(
            f"{self.title}",
            total=max(self.total_files, 1),
            completed=0,
            unit="files",
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._current_id is not None:
            self.progress.remove_task(self._current_id)
            self._current_id = None
        self.progress.stop()

    def skip(self, *, advance_overall: bool = True) -> None:
        if advance_overall:
            self.complete_file()

    def complete_file(self) -> None:
        with self._lock:
            self._advance_overall()

    def begin_blob_batch(self, label: str, blob_count: int, total_bytes: int) -> None:
        """One progress row for many blob uploads within a single indexed file."""
        with self._lock:
            if self._current_id is not None:
                self.progress.remove_task(self._current_id)
            self._aggregate_label = label
            self._blobs_done = 0
            self._blobs_total = max(int(blob_count), 1)
            display = self._blob_batch_description()
            total = max(int(total_bytes), 1)
            self._current_id = self.progress.add_task(
                display,
                total=total,
                unit="bytes",
                blobs_done=0,
                blobs_total=self._blobs_total,
            )

    def _blob_batch_description(self) -> str:
        name = self._aggregate_label if len(self._aggregate_label) <= 28 else f"{self._aggregate_label[:25]}..."
        return f"{name} · blobs {self._blobs_done}/{self._blobs_total}"

    def _mark_blob_done(self, *, advance_bytes: int = 0) -> None:
        if self._current_id is None:
            return
        self._blobs_done = min(self._blobs_done + 1, self._blobs_total)
        self.progress.update(
            self._current_id,
            advance=max(int(advance_bytes), 0),
            description=self._blob_batch_description(),
            blobs_done=self._blobs_done,
        )

    def on_aggregate_progress(self, bytes_amount: int) -> None:
        with self._lock:
            if self._current_id is None:
                return
            self.progress.update(self._current_id, advance=bytes_amount)

    def on_aggregate_blob_skip(self, byte_length: int) -> None:
        with self._lock:
            self._mark_blob_done(advance_bytes=byte_length)

    def finish_blob_batch(self) -> None:
        with self._lock:
            if self._current_id is not None:
                task = self.progress.tasks[
                    next(i for i, t in enumerate(self.progress.tasks) if t.id == self._current_id)
                ]
                self.progress.update(self._current_id, completed=task.total)
                self.progress.remove_task(self._current_id)
                self._current_id = None
            self._blobs_done = 0
            self._blobs_total = 0

    def on_blob_uploaded(self) -> None:
        with self._lock:
            self._mark_blob_done()

    def on_start(self, label: str, total_bytes: int) -> None:
        with self._lock:
            if self._current_id is not None:
                self.progress.remove_task(self._current_id)
            display = label if len(label) <= 36 else f"{label[:33]}..."
            total = max(int(total_bytes), 1)
            self._current_id = self.progress.add_task(display, total=total, unit="bytes")

    def on_progress(self, bytes_amount: int) -> None:
        with self._lock:
            if self._current_id is None:
                return
            self.progress.update(self._current_id, advance=bytes_amount)

    def on_finish(self, *, advance_overall: bool = True) -> None:
        with self._lock:
            if self._current_id is not None:
                task = self.progress.tasks[
                    next(i for i, t in enumerate(self.progress.tasks) if t.id == self._current_id)
                ]
                self.progress.update(self._current_id, completed=task.total)
                self.progress.remove_task(self._current_id)
                self._current_id = None
        if advance_overall:
            self.complete_file()

    def _advance_overall(self) -> None:
        self._files_done += 1
        if self._overall_id is not None:
            self.progress.update(
                self._overall_id,
                completed=min(self._files_done, max(self.total_files, 1)),
            )


def file_size(path: Path) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0
