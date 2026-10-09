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
    assert result.manifest["part_digest"] == "fast_range_v1"
    assert len(result.blobs) == 2
    blobs = {blob.content_hash: blob.read_payload() for blob in result.blobs}
    restored = tmp_path / "out.safetensors"
    handler.restore(result.manifest, blobs, restored)
    assert restored.read_bytes() == original
    round2 = handler.process(restored)
    assert round2.content_id == result.content_id
    assert round2.manifest == result.manifest


def _load_tensors_and_metadata(path: Path) -> tuple[dict | None, dict[str, np.ndarray]]:
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        metadata = handle.metadata()
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    return metadata, tensors


def test_restore_semantic_match_with_hf_metadata(tmp_path):
    """Header bytes may differ; safetensors.load semantics must match."""
    path = tmp_path / "hf.safetensors"
    save_file(
        {
            "weight": np.arange(6, dtype=np.float32).reshape(2, 3),
            "bias": np.array([1.0, 2.0], dtype=np.float16),
        },
        path,
        metadata={"format": "pt"},
    )
    handler = SafetensorsHandler()
    result = handler.process(path)
    restored = tmp_path / "restored.safetensors"
    blobs = {b.content_hash: b.read_payload() for b in result.blobs}
    handler.restore(result.manifest, blobs, restored)

    meta_src, tensors_src = _load_tensors_and_metadata(path)
    meta_out, tensors_out = _load_tensors_and_metadata(restored)
    assert meta_src == meta_out == {"format": "pt"}
    assert set(tensors_src) == set(tensors_out)
    for key in tensors_src:
        assert np.array_equal(tensors_src[key], tensors_out[key])
    assert handler.verify_local(restored, result.content_id, result.manifest)
    assert path.read_bytes() != restored.read_bytes()


def test_materialize_semantic_roundtrip(tmp_path):
    path = tmp_path / "m.safetensors"
    _write_demo(path)
    meta = build_file_metadata(path, Path("m.safetensors"))
    processed = SafetensorsHandler().process(path)
    blobs = {b.content_hash: b.read_payload() for b in processed.blobs}

    remote = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    remote.s3_bucket = "bucket"
    remote.s3_client = remote

    def fake_download(bucket, key, filename, Callback=None):
        Path(filename).write_bytes(blobs[key])

    remote.s3_client.download_file = fake_download
    remote.s3_client.head_object = lambda **kwargs: {
        "ContentLength": len(blobs[kwargs["Key"]])
    }

    dest = tmp_path / "pulled" / "m.safetensors"
    from express.base.storage.transfer_ops import materialize_local_file

    materialize_local_file(remote, meta, dest, force=True)
    _, tensors_src = _load_tensors_and_metadata(path)
    _, tensors_dest = _load_tensors_and_metadata(dest)
    for key in tensors_src:
        assert np.array_equal(tensors_src[key], tensors_dest[key])


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


def test_process_skips_safetensors_metadata_key(tmp_path):
    path = tmp_path / "meta.safetensors"
    save_file(
        {"weight": np.array([1.0, 2.0], dtype=np.float32)},
        path,
        metadata={"format": "pt"},
    )
    handler = SafetensorsHandler()
    result = handler.process(path)
    assert len(result.blobs) == 1
    assert result.manifest["tensor_order"] == ["weight"]
    assert "__metadata__" in result.manifest["header"]
    restored = tmp_path / "out.safetensors"
    blobs = {b.content_hash: b.read_payload() for b in result.blobs}
    handler.restore(result.manifest, blobs, restored)
    round2 = handler.process(restored)
    assert round2.content_id == result.content_id


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
