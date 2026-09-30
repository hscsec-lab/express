from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from express.base.storage.base import ProcessResult, StorageBlob, StorageUnitHandler, sha256_bytes

STORAGE_UNIT = "safetensors_v1"


def _read_safetensors_layout(path: Path) -> tuple[dict[str, Any], list[str], list[bytes]]:
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"not a safetensors file: {path}")
    header_size = struct.unpack("<Q", raw[:8])[0]
    header_end = 8 + header_size
    if header_end > len(raw):
        raise ValueError(f"truncated safetensors header: {path}")
    header: dict[str, Any] = json.loads(raw[8:header_end].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header: {path}")
    data_base = header_end
    order = list(header.keys())
    blobs: list[bytes] = []
    for name in order:
        info = header[name]
        start, end = info["data_offsets"]
        blobs.append(raw[data_base + start : data_base + end])
    return header, order, blobs


def _compose_safetensors(header: dict[str, Any], order: list[str], blobs: dict[str, bytes]) -> bytes:
    offset = 0
    rebuilt: dict[str, Any] = {}
    for name in order:
        if name not in header:
            raise KeyError(f"missing tensor header for {name!r}")
        chunk = blobs[name]
        entry = {k: v for k, v in header[name].items() if k != "data_offsets"}
        entry["data_offsets"] = [offset, offset + len(chunk)]
        rebuilt[name] = entry
        offset += len(chunk)
    header_bytes = json.dumps(rebuilt, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    body = b"".join(blobs[name] for name in order)
    return struct.pack("<Q", len(header_bytes)) + header_bytes + body


def manifest_fingerprint(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


class SafetensorsHandler(StorageUnitHandler):
    kind = STORAGE_UNIT
    priority = 100

    @classmethod
    def matches(cls, path: Path) -> bool:
        if path.suffix.lower() != ".safetensors" or not path.is_file():
            return False
        try:
            _read_safetensors_layout(path)
            return True
        except (ValueError, json.JSONDecodeError, KeyError, TypeError, OSError):
            return False

    def process(self, path: Path) -> ProcessResult:
        header, order, raw_blobs = _read_safetensors_layout(path)
        parts: list[dict[str, Any]] = []
        storage_blobs: list[StorageBlob] = []
        blob_map: dict[str, bytes] = {}
        for name, chunk in zip(order, raw_blobs, strict=True):
            content_hash = sha256_bytes(chunk)
            parts.append({"name": name, "content_hash": content_hash, "byte_length": len(chunk)})
            blob_map[content_hash] = chunk
            storage_blobs.append(StorageBlob(content_hash=content_hash, data=chunk))
        manifest = {
            "storage_unit": STORAGE_UNIT,
            "header": header,
            "tensor_order": order,
            "parts": parts,
        }
        return ProcessResult(
            storage_unit=self.kind,
            content_id=manifest_fingerprint(manifest),
            manifest=manifest,
            blobs=storage_blobs,
        )

    def restore(self, manifest: dict, blobs: dict[str, bytes], dest: Path) -> None:
        if manifest.get("storage_unit") != STORAGE_UNIT:
            raise ValueError("not a safetensors_v1 manifest")
        header = manifest["header"]
        order = manifest["tensor_order"]
        named: dict[str, bytes] = {}
        for part in manifest["parts"]:
            name = part["name"]
            content_hash = part["content_hash"]
            if content_hash not in blobs:
                raise KeyError(f"missing blob for tensor {name!r} ({content_hash})")
            chunk = blobs[content_hash]
            if len(chunk) != part["byte_length"]:
                raise ValueError(f"length mismatch for tensor {name!r}")
            named[name] = chunk
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_compose_safetensors(header, order, named))
