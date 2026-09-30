"""
Safetensors storage: split/merge tensor blobs for sync.

Round-trip goal is **semantic** equivalence (same tensor names, dtypes, shapes, payload
bytes, and ``__metadata__`` when present) as loaded by the official ``safetensors``
library. Rebuilt files are **not** guaranteed to be byte-identical to the original:
header JSON may differ in length, key order, or spacing; tensor data is re-packed
contiguously in ``tensor_order``.
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path
from typing import Any, Callable

from express.base.file import fast_range_digest
from express.base.storage.base import ProcessResult, StorageBlob, StorageUnitHandler, sha256_bytes

STORAGE_UNIT = "safetensors_v1"
PART_DIGEST_FAST = "fast_range_v1"
PART_DIGEST_FULL = "sha256_full"
SAFETENSORS_METADATA_KEY = "__metadata__"


def _is_tensor_entry(entry: Any) -> bool:
    return isinstance(entry, dict) and "data_offsets" in entry


def _tensor_names_from_header(header: dict[str, Any]) -> list[str]:
    """Tensor keys only (excludes ``__metadata__`` and other non-tensor header entries)."""
    return [name for name in header if _is_tensor_entry(header[name])]


def _parse_header(raw: memoryview) -> tuple[dict[str, Any], list[str], int]:
    if len(raw) < 8:
        raise ValueError("not a safetensors file")  # pragma: no cover
    header_size = struct.unpack("<Q", raw[:8])[0]
    header_end = 8 + header_size
    if header_end > len(raw):
        raise ValueError("truncated safetensors header")  # pragma: no cover
    header: dict[str, Any] = json.loads(raw[8:header_end].tobytes().decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("invalid safetensors header")  # pragma: no cover
    return header, list(header.keys()), header_end


def _read_safetensors_layout(path: Path) -> tuple[dict[str, Any], list[str], list[bytes]]:
    """Load all tensor bytes (small files / tests only)."""
    raw = path.read_bytes()
    header, _, header_end = _parse_header(memoryview(raw))
    order = _tensor_names_from_header(header)
    data_base = header_end
    blobs: list[bytes] = []
    for name in order:
        start, end = header[name]["data_offsets"]
        blobs.append(raw[data_base + start : data_base + end])
    return header, order, blobs


def _sha256_mmap_range(mm: mmap.mmap, start: int, length: int, chunk_size: int = 8 * 1024 * 1024) -> str:
    import hashlib

    hasher = hashlib.sha256()
    pos = start
    end = start + length
    while pos < end:
        chunk_end = min(pos + chunk_size, end)
        hasher.update(mm[pos:chunk_end])
        pos = chunk_end
    return hasher.hexdigest()


def _part_content_hash(mm: mmap.mmap, start: int, length: int, digest_mode: str) -> str:
    if digest_mode == PART_DIGEST_FAST:
        return fast_range_digest(mm, start, length)
    if digest_mode == PART_DIGEST_FULL:
        return _sha256_mmap_range(mm, start, length)
    raise ValueError(f"unknown part digest mode: {digest_mode!r}")


def _compose_safetensors(header: dict[str, Any], order: list[str], blobs: dict[str, bytes]) -> bytes:
    offset = 0
    rebuilt: dict[str, Any] = {}
    if SAFETENSORS_METADATA_KEY in header:
        rebuilt[SAFETENSORS_METADATA_KEY] = header[SAFETENSORS_METADATA_KEY]
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
            with path.open("rb") as handle:
                prefix = handle.read(8)
                if len(prefix) < 8:
                    return False
                header_size = struct.unpack("<Q", prefix)[0]
                if header_size > 32 * 1024 * 1024:
                    return False
                header_bytes = handle.read(header_size)
                if len(header_bytes) != header_size:  # pragma: no cover
                    return False
            json.loads(header_bytes.decode("utf-8"))
            return True
        except (ValueError, json.JSONDecodeError, KeyError, TypeError, OSError, struct.error):  # pragma: no cover
            return False

    def process(
            self,
            path: Path,
            *,
            on_tensor: Callable[[str, int, int], None] | None = None,
    ) -> ProcessResult:
        parts: list[dict[str, Any]] = []
        storage_blobs: list[StorageBlob] = []

        with path.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                header, _, data_base = _parse_header(memoryview(mm))
                order = _tensor_names_from_header(header)
                total = len(order)
                for index, name in enumerate(order, start=1):
                    start, end = header[name]["data_offsets"]
                    abs_start = data_base + start
                    byte_length = end - start
                    if on_tensor is not None:
                        on_tensor(name, index, total)
                    content_hash = _part_content_hash(
                        mm, abs_start, byte_length, PART_DIGEST_FAST
                    )
                    parts.append({"name": name, "content_hash": content_hash, "byte_length": byte_length})
                    storage_blobs.append(
                        StorageBlob(
                            content_hash=content_hash,
                            source_path=path,
                            byte_offset=abs_start,
                            byte_length=byte_length,
                        )
                    )

        manifest = {
            "storage_unit": STORAGE_UNIT,
            "part_digest": PART_DIGEST_FAST,
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

    def verify_local(self, path: Path, content_id: str, manifest: dict | None) -> bool:
        if manifest is None:
            return self.process(path).content_id == content_id
        computed = manifest_fingerprint(manifest)
        if computed != content_id:
            return False
        digest_mode = manifest.get("part_digest", PART_DIGEST_FULL)
        with path.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                file_header, _, data_base = _parse_header(memoryview(mm))
                for part in manifest["parts"]:
                    name = part["name"]
                    if name not in file_header:
                        return False  # pragma: no cover
                    start, end = file_header[name]["data_offsets"]
                    abs_start = data_base + start
                    byte_length = end - start
                    if byte_length != part["byte_length"]:  # pragma: no cover
                        return False
                    computed = _part_content_hash(mm, abs_start, byte_length, digest_mode)
                    if computed != part["content_hash"]:  # pragma: no cover
                        return False
        return True
