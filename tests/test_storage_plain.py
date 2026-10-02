from pathlib import Path

import pytest

from express.base.storage.plain import PlainFileHandler


def test_plain_restore_errors(tmp_path):
    handler = PlainFileHandler()
    dest = tmp_path / "out.bin"
    with pytest.raises(ValueError):
        handler.restore(None, {}, dest)
    with pytest.raises(ValueError):
        handler.restore(None, {"a": b"1", "b": b"2"}, dest)


def test_plain_roundtrip(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"payload")
    handler = PlainFileHandler()
    result = handler.process(src)
    dest = tmp_path / "b.bin"
    handler.restore(None, {result.content_id: result.blobs[0].read_payload()}, dest)
    assert dest.read_bytes() == b"payload"
