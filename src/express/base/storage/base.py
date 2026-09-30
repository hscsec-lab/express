from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file_range(path: Path, offset: int, length: int, chunk_size: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(offset)
        remaining = length
        while remaining > 0:
            block = handle.read(min(chunk_size, remaining))
            if not block:
                break
            hasher.update(block)
            remaining -= len(block)
    return hasher.hexdigest()


class StorageBlob(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    content_hash: str
    data: bytes = Field(default=b"", repr=False)
    source_path: Path | None = None
    byte_offset: int = 0
    byte_length: int = 0

    @property
    def payload_length(self) -> int:
        if self.source_path is not None and self.byte_length:
            return self.byte_length
        return len(self.data)

    def read_payload(self) -> bytes:
        if self.data:
            return self.data
        if self.source_path is not None and self.byte_length:
            with self.source_path.open("rb") as handle:
                handle.seek(self.byte_offset)
                return handle.read(self.byte_length)
        return b""


class ProcessResult(BaseModel):
    storage_unit: str
    content_id: str
    manifest: dict | None = None
    blobs: list[StorageBlob] = Field(default_factory=list)

    def iter_blobs(self) -> Iterator[StorageBlob]:
        yield from self.blobs


class StorageUnitHandler(ABC):
    """Decompose a logical file into content-addressed blobs and restore it."""

    kind: str
    priority: int = 0

    @classmethod
    @abstractmethod
    def matches(cls, path: Path) -> bool:  # pragma: no cover
        ...

    @abstractmethod
    def process(self, path: Path) -> ProcessResult:
        """Split *path* into blobs + manifest; *content_id* identifies the logical file."""

    @abstractmethod
    def restore(self, manifest: dict, blobs: dict[str, bytes], dest: Path) -> None:
        """Write *dest* from *manifest* and hash→bytes map."""

    def verify_local(self, path: Path, content_id: str, manifest: dict | None) -> bool:
        fresh = self.process(path)
        if fresh.content_id != content_id:
            return False
        if manifest is not None and fresh.manifest != manifest:
            return False
        return True
