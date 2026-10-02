from pathlib import Path

from express.base.storage.base import StorageBlob, sha256_file_range
from express.base.storage.plain import PlainFileHandler


def test_read_payload_empty():
    assert StorageBlob(content_hash="x").read_payload() == b""


def test_sha256_file_range_truncated(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert sha256_file_range(path, 0, 100) == sha256_file_range(path, 0, 3)


def test_plain_verify_local_ok(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"ok")
    handler = PlainFileHandler()
    result = handler.process(path)
    assert handler.verify_local(path, result.content_id, None) is True


def test_verify_local_mismatch(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"one")
    handler = PlainFileHandler()
    assert handler.verify_local(path, "wrong-id", None) is False
    result = handler.process(path)
    assert handler.verify_local(path, result.content_id, {"unexpected": True}) is False
    assert next(iter(result.iter_blobs())).content_hash


def test_process_on_tensor_callback(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    from express.base.storage.safetensors import SafetensorsHandler

    path = tmp_path / "t.safetensors"
    save_file({"a": np.ones(2, dtype=np.float32), "b": np.ones(3, dtype=np.float32)}, path)
    seen: list[str] = []
    SafetensorsHandler().process(path, on_tensor=lambda name, i, t: seen.append(name))
    assert seen == ["a", "b"]
