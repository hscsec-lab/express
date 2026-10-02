from pathlib import Path
from unittest.mock import patch

from express.base.progress import (
    TransferSession,
    busy_status,
    catalog_progress,
    file_size,
    index_progress,
    short_label,
)


def test_transfer_session_lifecycle(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 10)
    with TransferSession("Upload", total_files=1) as session:
        session.on_start("f.bin", file_size(path))
        session.on_progress(5)
        session.on_finish()
        session.skip()


def test_busy_status_and_catalog_progress():
    with busy_status("working"):
        pass
    progress = catalog_progress()
    task = progress.add_task("scan", total=2)
    progress.advance(task)
    progress.stop()


def test_short_label_and_index_progress():
    long_name = "model.language_model.layers.32.mlp.experts.down_proj.weight"
    assert short_label(long_name, 20).startswith("…")
    assert len(short_label(long_name, 20)) == 20
    assert short_label("hello", 1) == "h"
    assert short_label("hello", 2) == "…o"
    progress = index_progress()
    task = progress.add_task("Indexing", total=1, tensor="")
    progress.update(task, description="Hash shard", tensor=f"{short_label(long_name, 30)} (1/10)")
    progress.stop()


def test_file_size_missing():
    assert file_size(Path("/nonexistent/file.bin")) == 0


def test_transfer_session_replace_current_task(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"a")
    with TransferSession("Up", total_files=2) as session:
        session.on_start("a.bin", 1)
        session.on_start("b.bin", 1)


def test_transfer_session_on_progress_without_start():
    session = TransferSession("X", total_files=0)
    session.progress.start()
    session.on_progress(10)
    session.progress.stop()


def test_transfer_session_exit_removes_current_task(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"x")
    with TransferSession("Up", total_files=1) as session:
        session.on_start("f.bin", 1)
        session.on_progress(1)
