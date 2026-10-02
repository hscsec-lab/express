import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from express.base.structure_scan import scan_model_structure


def test_scan_single_shard(tmp_path):
    save_file({"layer.weight": np.zeros((4, 8), dtype=np.float32)}, tmp_path / "model.safetensors")
    rows = scan_model_structure(tmp_path)
    assert rows["layer.weight"]["params"] == 32
    assert rows["layer.weight"]["shape"] == [4, 8]


def test_scan_index_json_shards(tmp_path):
    save_file({"a": np.ones(2, dtype=np.float32)}, tmp_path / "a.safetensors")
    save_file({"b": np.ones(3, dtype=np.float32)}, tmp_path / "b.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}),
        encoding="utf-8",
    )
    rows = scan_model_structure(tmp_path)
    assert set(rows) == {"a", "b"}


def test_scan_skips_metadata(tmp_path):
    save_file({"w": np.array([1.0], dtype=np.float32)}, tmp_path / "m.safetensors", metadata={"format": "pt"})
    rows = scan_model_structure(tmp_path)
    assert list(rows) == ["w"]


def test_scan_no_weights_raises(tmp_path):
    with pytest.raises(RuntimeError, match="No .safetensors"):
        scan_model_structure(tmp_path)


def test_read_header_errors(tmp_path):
    import struct

    from express.base.structure_scan import _read_safetensors_header

    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"short")
    with pytest.raises(ValueError):
        _read_safetensors_header(bad)

    huge = tmp_path / "huge.safetensors"
    huge.write_bytes(struct.pack("<Q", 33 * 1024 * 1024))
    with pytest.raises(ValueError, match="too large"):
        _read_safetensors_header(huge)

    truncated = tmp_path / "trunc.safetensors"
    truncated.write_bytes(struct.pack("<Q", 10) + b"12345")
    with pytest.raises(ValueError, match="truncated"):
        _read_safetensors_header(truncated)

    not_dict = tmp_path / "list.safetensors"
    payload = json.dumps([1, 2]).encode("utf-8")
    not_dict.write_bytes(struct.pack("<Q", len(payload)) + payload)
    with pytest.raises(ValueError, match="invalid"):
        _read_safetensors_header(not_dict)


def test_duplicate_tensor_raises(tmp_path):
    save_file({"x": np.ones(1, dtype=np.float32)}, tmp_path / "a.safetensors")
    save_file({"x": np.ones(2, dtype=np.float32)}, tmp_path / "b.safetensors")
    with pytest.raises(ValueError, match="duplicate"):
        scan_model_structure(tmp_path)


def test_empty_tensor_header(tmp_path):
    import struct

    header = json.dumps({"__metadata__": {"format": "pt"}}).encode("utf-8")
    path = tmp_path / "empty.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(RuntimeError, match="No tensors"):
        scan_model_structure(tmp_path)
