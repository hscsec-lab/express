from pathlib import Path

from express.base.progress import TransferSession, file_size


def test_file_size(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"12345")
    assert file_size(path) == 5
    assert file_size(tmp_path / "missing.bin") == 0


def test_transfer_session_advances_overall(tmp_path):
    session = TransferSession("Push", total_files=2)
    with session:
        cb = session.begin_file("a.bin", 10)
        cb(4)
        cb(6)
        session.finish_file()
        session.skip()
    assert session._files_done == 2
