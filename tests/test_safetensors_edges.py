import json
import struct
from pathlib import Path

import pytest

from express.base.storage.safetensors import SafetensorsHandler, _compose_safetensors


def test_compose_missing_tensor():
    header = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    with pytest.raises(KeyError):
        _compose_safetensors(header, ["a"], {})


def test_restore_bad_manifest(tmp_path):
    handler = SafetensorsHandler()
    with pytest.raises(ValueError):
        handler.restore({"storage_unit": "plain"}, {}, tmp_path / "x.safetensors")


def test_truncated_file(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00")
    assert SafetensorsHandler.matches(path) is False


def test_restore_length_mismatch(tmp_path):
    handler = SafetensorsHandler()
    manifest = {
        "storage_unit": "safetensors_v1",
        "header": {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        "tensor_order": ["w"],
        "parts": [{"name": "w", "content_hash": "abc", "byte_length": 4}],
    }
    with pytest.raises(ValueError):
        handler.restore(manifest, {"abc": b"123"}, tmp_path / "o.safetensors")
