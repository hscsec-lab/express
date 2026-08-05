from express.base.progress import TransferSession, file_size


def test_file_size(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"12345")
    assert file_size(path) == 5
    assert file_size(tmp_path / "missing.bin") == 0


def test_transfer_session_advances_overall():
    session = TransferSession("Push", total_files=2)
    with session:
        session.on_start("a.bin", 10)
        session.on_progress(4)
        session.on_progress(6)
        session.on_finish()
        session.skip()
    assert session._files_done == 2
