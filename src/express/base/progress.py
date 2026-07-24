"""Modern terminal progress helpers built on Rich."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from express import console


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


class TransferSession:
    """
    Single live progress view for multi-file push/pull.

    Shows overall file completion plus the active file's byte progress.
    """

    def __init__(self, title: str, total_files: int):
        self.title = title
        self.total_files = max(total_files, 0)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=28),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=True,
        )
        self._overall_id = None
        self._current_id = None
        self._files_done = 0

    def __enter__(self) -> "TransferSession":
        self.progress.start()
        self._overall_id = self.progress.add_task(
            f"{self.title}",
            total=max(self.total_files, 1),
            completed=0,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._current_id is not None:
            self.progress.remove_task(self._current_id)
            self._current_id = None
        self.progress.stop()

    def skip(self) -> None:
        self._advance_overall()

    def begin_file(self, label: str, total_bytes: int) -> Callable[[int], None]:
        if self._current_id is not None:
            self.progress.remove_task(self._current_id)
        display = label if len(label) <= 36 else f"{label[:33]}..."
        total = max(int(total_bytes), 1)
        self._current_id = self.progress.add_task(display, total=total)
        task_id = self._current_id

        def callback(bytes_amount: int) -> None:
            self.progress.update(task_id, advance=bytes_amount)

        return callback

    def finish_file(self) -> None:
        if self._current_id is not None:
            task = self.progress.tasks[
                next(i for i, t in enumerate(self.progress.tasks) if t.id == self._current_id)
            ]
            self.progress.update(self._current_id, completed=task.total)
            self.progress.remove_task(self._current_id)
            self._current_id = None
        self._advance_overall()

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
