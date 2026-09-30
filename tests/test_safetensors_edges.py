import json
import struct
from pathlib import Path

import pytest

from express.base.storage.safetensors import (
    PART_DIGEST_FULL,
    SafetensorsHandler,
    _compose_safetensors,
    _part_content_hash,
    manifest_fingerprint,
)


def test_compose_missing_tensor():
    header = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    with pytest.raises(KeyError):
        _compose_safetensors(header, ["a"], {})


def test_restore_bad_manifest(tmp_path):
    handler = SafetensorsHandler()
    with pytest.raises(ValueError):
        handler.restore({"storage_unit": "plain"}, {}, tmp_path / "x.safetensors")


def test_read_layout_helper_small_file(tmp_path):
    from express.base.storage.safetensors import _read_safetensors_layout
    from safetensors.numpy import save_file
    import numpy as np

    path = tmp_path / "t.safetensors"
    save_file({"x": np.array([1.0], dtype=np.float32)}, path)
    header, order, blobs = _read_safetensors_layout(path)
    assert "x" in header
    assert len(blobs) == 1


def test_parse_header_rejects_short_buffer():
    import struct

    from express.base.storage.safetensors import _parse_header

    with pytest.raises(ValueError):
        _parse_header(memoryview(b""))


def test_matches_rejects_invalid_json_header(tmp_path):
    import struct

    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", 5) + b"{bad}")
    assert SafetensorsHandler.matches(path) is False


def test_truncated_file(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00")
    assert SafetensorsHandler.matches(path) is False


def test_part_digest_full_verify_legacy_manifest(tmp_path):
    import mmap

    import numpy as np
    from safetensors.numpy import save_file

    path = tmp_path / "t.safetensors"
    save_file({"w": np.arange(4, dtype=np.float32)}, path)
    handler = SafetensorsHandler()
    processed = handler.process(path)
    manifest = dict(processed.manifest)
    manifest.pop("part_digest", None)
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            from express.base.storage.safetensors import _parse_header

            _, _, data_base = _parse_header(memoryview(mm))
            header = manifest["header"]
            full_parts = []
            for part in manifest["parts"]:
                name = part["name"]
                start, end = header[name]["data_offsets"]
                abs_start = data_base + start
                byte_length = end - start
                full_parts.append(
                    {
                        **part,
                        "content_hash": _part_content_hash(
                            mm, abs_start, byte_length, PART_DIGEST_FULL
                        ),
                    }
                )
    manifest["parts"] = full_parts
    assert handler.verify_local(path, processed.content_id, manifest) is False
    manifest["part_digest"] = PART_DIGEST_FULL
    assert handler.verify_local(path, manifest_fingerprint(manifest), manifest) is True


def test_part_content_hash_rejects_unknown_mode(tmp_path):
    import mmap

    path = tmp_path / "x.bin"
    path.write_bytes(b"data")
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            with pytest.raises(ValueError):
                _part_content_hash(mm, 0, 4, "unknown")


def test_verify_without_manifest(tmp_path):
    from safetensors.numpy import save_file
    import numpy as np

    path = tmp_path / "t.safetensors"
    save_file({"w": np.array([1.0], dtype=np.float32)}, path)
    handler = SafetensorsHandler()
    result = handler.process(path)
    assert handler.verify_local(path, result.content_id, None) is True


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
