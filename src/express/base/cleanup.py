"""Cooperative cleanup when transfers are interrupted (e.g. Ctrl-C)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, TYPE_CHECKING

from express import console

if TYPE_CHECKING:
    from express.base.client import Remote

CleanupFn = Callable[[], None]


class CleanupScope:
    """
    Collects in-flight artifacts during push/pull.

    Handlers register partial local paths and remote keys; ``run()`` removes them.
    Extra hooks via ``register()`` for future operations (multipart, temp dirs, etc.).
    """

    def __init__(self) -> None:
        self._partial_locals: set[Path] = set()
        self._partial_remote: set[tuple[Remote, str]] = set()
        self._hooks: list[CleanupFn] = []

    def register(self, hook: CleanupFn) -> None:
        self._hooks.append(hook)

    def track_local_file(self, path: Path) -> None:
        self._partial_locals.add(Path(path))

    def clear_local_file(self, path: Path) -> None:
        self._partial_locals.discard(Path(path))

    def track_remote_key(self, remote: Remote, key: str) -> None:
        self._partial_remote.add((remote, key))

    def clear_remote_key(self, remote: Remote, key: str) -> None:
        self._partial_remote.discard((remote, key))

    def run(self) -> list[str]:
        """Run all cleanups; return human-readable log lines."""
        lines: list[str] = []
        for hook in reversed(self._hooks):
            hook()
        for remote, key in list(self._partial_remote):
            if _try_delete_remote(remote, key):
                lines.append(f"Removed incomplete remote object {key[:20]}…")
        for path in list(self._partial_locals):
            if path.exists():
                path.unlink(missing_ok=True)
                lines.append(f"Removed partial local file {path}")
        self._partial_locals.clear()
        self._partial_remote.clear()
        self._hooks.clear()
        return lines


def _try_delete_remote(remote: Remote, key: str) -> bool:
    try:
        remote.s3_client.delete_object(Bucket=remote.s3_bucket, Key=key)
        return True
    except Exception:  # pragma: no cover — best-effort cleanup
        return False


@contextmanager
def managed_transfer(operation: str) -> Iterator[CleanupScope]:
    """
    Wrap push/pull entrypoints: on KeyboardInterrupt, clean partial artifacts then exit 130.
    """
    scope = CleanupScope()
    try:
        yield scope
    except KeyboardInterrupt:
        for line in scope.run():
            console.print(f"[dim]{line}[/dim]")
        console.print(
            f"[yellow]{operation} interrupted[/yellow] (Ctrl-C). "
            "Partial uploads/downloads were cleaned up where possible."
        )
        # Do not raise SystemExit: non-daemon upload threads may still be running after
        # parallel blob push and would block process shutdown until S3 calls finish.
        os._exit(130)
