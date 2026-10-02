from __future__ import annotations

from pathlib import Path

from express.base.file import fast_checksum
from express.base.storage.base import ProcessResult, StorageBlob, StorageUnitHandler


class PlainFileHandler(StorageUnitHandler):
    kind = "plain"
    priority = 0

    @classmethod
    def matches(cls, path: Path) -> bool:
        return path.is_file()

    def process(self, path: Path, **kwargs) -> ProcessResult:
        content_id = fast_checksum(path)
        size = path.stat().st_size
        return ProcessResult(
            storage_unit=self.kind,
            content_id=content_id,
            manifest=None,
            blobs=[
                StorageBlob(
                    content_hash=content_id,
                    source_path=path,
                    byte_offset=0,
                    byte_length=size,
                )
            ],
        )

    def restore(self, manifest: dict | None, blobs: dict[str, bytes], dest: Path) -> None:
        if not blobs:
            raise ValueError("plain restore requires exactly one blob")
        if len(blobs) != 1:
            raise ValueError("plain restore expects a single blob")
        chunk = next(iter(blobs.values()))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(chunk)
