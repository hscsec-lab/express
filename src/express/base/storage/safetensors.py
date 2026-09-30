from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path
from typing import Any, Callable

from express.base.storage.base import ProcessResult, StorageBlob, StorageUnitHandler, sha256_bytes, sha256_file_range

STORAGE_UNIT = "safetensors_v1"


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
    header, order, header_end = _parse_header(memoryview(raw))
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
                header, order, data_base = _parse_header(memoryview(mm))
                total = len(order)
                for index, name in enumerate(order, start=1):
                    start, end = header[name]["data_offsets"]
                    abs_start = data_base + start
                    byte_length = end - start
                    if on_tensor is not None:
                        on_tensor(name, index, total)
                    content_hash = _sha256_mmap_range(mm, abs_start, byte_length)
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
        header = manifest["header"]
        order = manifest["tensor_order"]
        data_base = None
        with path.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                _, _, data_base = _parse_header(memoryview(mm))
                for part in manifest["parts"]:
                    name = part["name"]
                    start, end = header[name]["data_offsets"]
                    abs_start = data_base + start
                    byte_length = end - start
                    if byte_length != part["byte_length"]:  # pragma: no cover
                        return False
                    if _sha256_mmap_range(mm, abs_start, byte_length) != part["content_hash"]:  # pragma: no cover
                        return False
        return True
