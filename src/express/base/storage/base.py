from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class StorageBlob(BaseModel):
    content_hash: str
    data: bytes = Field(repr=False)

    @property
    def byte_length(self) -> int:
        return len(self.data)


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
