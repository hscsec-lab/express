import os
from unittest.mock import patch

from express.base.s3 import DownloadProgressSimple, ProgressBase, ProgressPercentage


def test_progress_base_format_bytes():
    base = ProgressBase(silent=True)
    assert "KB" in base._format_bytes(2048)


def test_download_progress_simple_callback():
    prog = DownloadProgressSimple(filename="demo.bin", silent=True)
    prog(1024)
    prog(1024)
    prog.done()


def test_progress_percentage(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 100)
    prog = ProgressPercentage(str(path), silent=True)
    prog(50)
    prog(50)
