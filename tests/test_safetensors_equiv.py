from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from express.base.storage.safetensors_equiv import (
    compare_safetensors_paths,
    print_whole_file_hash_hint,
)


def test_compare_equivalent_with_different_bytes(tmp_path):
    left = tmp_path / "a.safetensors"
    right = tmp_path / "b.safetensors"
    save_file({"w": np.array([1.0, 2.0], dtype=np.float32)}, left, metadata={"format": "pt"})
    save_file({"w": np.array([1.0, 2.0], dtype=np.float32)}, right)

    from express.base.storage.safetensors import SafetensorsHandler

    handler = SafetensorsHandler()
    result = handler.process(left)
    blobs = {b.content_hash: b.read_payload() for b in result.blobs}
    handler.restore(result.manifest, blobs, right)

    cmp_result = compare_safetensors_paths(left, right)
    assert cmp_result.equivalent is True
    assert cmp_result.bytes_identical is False
    assert cmp_result.differences == ()


def test_compare_detects_tensor_mismatch(tmp_path):
    left = tmp_path / "a.safetensors"
    right = tmp_path / "b.safetensors"
    save_file({"w": np.array([1.0], dtype=np.float32)}, left)
    save_file({"w": np.array([2.0], dtype=np.float32)}, right)
    result = compare_safetensors_paths(left, right)
    assert not result.equivalent
    assert any("values differ" in d for d in result.differences)


def test_compare_detects_missing_tensor(tmp_path):
    left = tmp_path / "a.safetensors"
    right = tmp_path / "b.safetensors"
    save_file({"a": np.array([1.0], dtype=np.float32), "b": np.array([2.0], dtype=np.float32)}, left)
    save_file({"a": np.array([1.0], dtype=np.float32)}, right)
    result = compare_safetensors_paths(left, right)
    assert not result.equivalent
    assert any("only in a.safetensors" in d for d in result.differences)

    save_file({"a": np.array([1.0], dtype=np.float32), "c": np.array([3.0], dtype=np.float32)}, right)
    result2 = compare_safetensors_paths(left, right)
    assert any("only in b.safetensors" in d for d in result2.differences)


def test_compare_rejects_non_safetensors(tmp_path):
    bad = tmp_path / "x.bin"
    bad.write_bytes(b"data")
    good = tmp_path / "a.safetensors"
    save_file({"w": np.array([1.0], dtype=np.float32)}, good)
    assert not compare_safetensors_paths(bad, good).equivalent
    assert not compare_safetensors_paths(good, bad).equivalent


def test_compare_dtype_and_shape(tmp_path):
    left = tmp_path / "a.safetensors"
    right = tmp_path / "b.safetensors"
    save_file({"w": np.array([1.0, 2.0], dtype=np.float32)}, left)
    save_file({"w": np.array([1, 2], dtype=np.int32)}, right)
    dtype_diff = compare_safetensors_paths(left, right)
    assert any("dtype" in d for d in dtype_diff.differences)

    save_file({"w": np.array([[1.0]], dtype=np.float32)}, right)
    shape_diff = compare_safetensors_paths(left, right)
    assert any("shape" in d for d in shape_diff.differences)


def test_compare_metadata_diff(tmp_path):
    left = tmp_path / "a.safetensors"
    right = tmp_path / "b.safetensors"
    save_file({"w": np.array([1.0], dtype=np.float32)}, left, metadata={"format": "pt"})
    save_file({"w": np.array([1.0], dtype=np.float32)}, right, metadata={"format": "mlx"})
    result = compare_safetensors_paths(left, right)
    assert not result.equivalent
    assert any("metadata" in d for d in result.differences)


def test_print_hint(capsys):
    print_whole_file_hash_hint(path=Path("model.safetensors"))
    assert "cmp-safetensors" in capsys.readouterr().out
