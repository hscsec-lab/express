from pathlib import Path

from express.base.storage.plain import PlainFileHandler


def test_verify_local_mismatch(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"one")
    handler = PlainFileHandler()
    assert handler.verify_local(path, "wrong-id", None) is False
