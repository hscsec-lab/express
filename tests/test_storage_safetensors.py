from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from express.base.storage.plain import PlainFileHandler
from express.base.storage.registry import build_file_metadata, content_keys_for_metadata, resolve_handler
from express.base.storage.safetensors import SafetensorsHandler, manifest_fingerprint


def _write_demo(path: Path) -> None:
    save_file(
        {
            "weight": np.arange(6, dtype=np.float32).reshape(2, 3),
            "bias": np.array([1.0, 2.0], dtype=np.float16),
        },
        path,
    )


def test_handler_priority_prefers_safetensors(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_demo(path)
    assert isinstance(resolve_handler(path), SafetensorsHandler)
    assert isinstance(resolve_handler(tmp_path / "readme.txt"), PlainFileHandler)


def test_process_restore_roundtrip(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_demo(path)
    original = path.read_bytes()
    handler = SafetensorsHandler()
    result = handler.process(path)
    assert result.storage_unit == "safetensors_v1"
    assert len(result.blobs) == 2
    blobs = {blob.content_hash: blob.read_payload() for blob in result.blobs}
    restored = tmp_path / "out.safetensors"
    handler.restore(result.manifest, blobs, restored)
    assert restored.read_bytes() == original
    round2 = handler.process(restored)
    assert round2.content_id == result.content_id
    assert round2.manifest == result.manifest


def test_modify_tensors_changes_only_some_hashes(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_demo(path)
    handler = SafetensorsHandler()
    before = handler.process(path)
    before_hashes = {p["name"]: p["content_hash"] for p in before.manifest["parts"]}

    tensors = {"weight": np.zeros((2, 3), dtype=np.float32), "bias": np.array([1.0, 2.0], dtype=np.float16)}
    save_file(tensors, path)
    after = handler.process(path)
    after_hashes = {p["name"]: p["content_hash"] for p in after.manifest["parts"]}

    assert before_hashes["bias"] == after_hashes["bias"]
    assert before_hashes["weight"] != after_hashes["weight"]
    assert before.content_id != after.content_id


def test_build_file_metadata_and_content_keys(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_demo(path)
    meta = build_file_metadata(path, Path("model.safetensors"))
    assert meta.storage_unit == "safetensors_v1"
    assert meta.unit_manifest is not None
    assert meta.file_checksum_sha256 == manifest_fingerprint(meta.unit_manifest)
    keys = content_keys_for_metadata(meta)
    assert len(keys) == 2
    assert keys == {p["content_hash"] for p in meta.unit_manifest["parts"]}


def test_invalid_safetensors_falls_back_to_plain(tmp_path):
    path = tmp_path / "fake.safetensors"
    path.write_bytes(b"not-safetensors")
    assert isinstance(resolve_handler(path), PlainFileHandler)


def test_restore_rejects_missing_blob(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_demo(path)
    handler = SafetensorsHandler()
    result = handler.process(path)
    with pytest.raises(KeyError):
        handler.restore(result.manifest, {}, tmp_path / "x.safetensors")
