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

_CHILD_PREFIX = "  └ "
_IDLE_CHILD = f"{_CHILD_PREFIX}[dim]·[/dim]"


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
    Safetensors shard uploads use a parent row (file) and indented child rows
    (one per parallel upload slot, each bound to a tensor while active).
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
        self._parent_file_id: int | None = None
        self._slot_ids: list[int] = []
        self._files_done = 0
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
        self._clear_simple_file_task()
        self._clear_decomposed_file_tasks()
        self.progress.stop()

    def skip(self, *, advance_overall: bool = True) -> None:
        if advance_overall:
            self.complete_file()

    def complete_file(self) -> None:
        with self._lock:
            self._advance_overall()

    def begin_decomposed_file(self, label: str, total_bytes: int, *, slot_count: int) -> None:
        """Parent file row plus fixed parallel slot rows for tensor uploads."""
        with self._lock:
            self._clear_simple_file_task()
            self._clear_decomposed_file_tasks()
            display = label if len(label) <= 36 else f"{label[:33]}..."
            total = max(int(total_bytes), 1)
            self._parent_file_id = self.progress.add_task(
                display,
                total=total,
                unit="bytes",
            )
            slots = max(int(slot_count), 1)
            self._slot_ids = []
            for _ in range(slots):
                slot_id = self.progress.add_task(
                    _IDLE_CHILD,
                    total=1,
                    completed=0,
                    unit="bytes",
                    nested=True,
                )
                self._slot_ids.append(slot_id)

    def on_slot_start(self, slot: int, tensor_name: str, total_bytes: int) -> None:
        with self._lock:
            if slot < 0 or slot >= len(self._slot_ids):
                return
            total = max(int(total_bytes), 1)
            desc = f"{_CHILD_PREFIX}{short_label(tensor_name, 44)}"
            self.progress.update(
                self._slot_ids[slot],
                description=desc,
                total=total,
                completed=0,
            )

    def on_slot_progress(self, slot: int, bytes_amount: int) -> None:
        with self._lock:
            if slot < 0 or slot >= len(self._slot_ids):
                return
            advance = max(int(bytes_amount), 0)
            if advance:
                self.progress.update(self._slot_ids[slot], advance=advance)
                if self._parent_file_id is not None:
                    self.progress.update(self._parent_file_id, advance=advance)

    def on_slot_skip(self, slot: int, byte_length: int) -> None:
        with self._lock:
            if slot < 0 or slot >= len(self._slot_ids):
                return
            total = max(int(byte_length), 1)
            self.progress.update(self._slot_ids[slot], total=total, completed=total)
            if self._parent_file_id is not None:
                self.progress.update(self._parent_file_id, advance=byte_length)
            self._reset_slot(slot)

    def on_slot_done(self, slot: int) -> None:
        with self._lock:
            self._reset_slot(slot)

    def _reset_slot(self, slot: int) -> None:
        if slot < 0 or slot >= len(self._slot_ids):
            return
        self.progress.update(
            self._slot_ids[slot],
            description=_IDLE_CHILD,
            total=1,
            completed=0,
        )

    def finish_decomposed_file(self) -> None:
        with self._lock:
            if self._parent_file_id is not None:
                task = self._task_by_id(self._parent_file_id)
                if task is not None:
                    self.progress.update(self._parent_file_id, completed=task.total)
            self._clear_decomposed_file_tasks()

    def on_start(self, label: str, total_bytes: int) -> None:
        with self._lock:
            self._clear_decomposed_file_tasks()
            self._clear_simple_file_task()
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
            self._clear_simple_file_task()
        if advance_overall:
            self.complete_file()

    def _clear_simple_file_task(self) -> None:
        if self._current_id is None:
            return
        task = self._task_by_id(self._current_id)
        if task is not None:
            self.progress.update(self._current_id, completed=task.total)
        self.progress.remove_task(self._current_id)
        self._current_id = None

    def _clear_decomposed_file_tasks(self) -> None:
        for slot_id in self._slot_ids:
            self.progress.remove_task(slot_id)
        self._slot_ids = []
        if self._parent_file_id is not None:
            self.progress.remove_task(self._parent_file_id)
            self._parent_file_id = None

    def _task_by_id(self, task_id: int):
        for task in self.progress.tasks:
            if task.id == task_id:
                return task
        return None

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
