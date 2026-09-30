from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set, Type

from express.base.file import FileMetadata
from express.base.storage.base import StorageUnitHandler
from express.base.storage.plain import PlainFileHandler
from express.base.storage.safetensors import SafetensorsHandler

_HANDLERS: tuple[Type[StorageUnitHandler], ...] = (
    SafetensorsHandler,
    PlainFileHandler,
)


def resolve_handler(path: Path) -> StorageUnitHandler:
    for handler_cls in sorted(_HANDLERS, key=lambda cls: -cls.priority):
        if handler_cls.matches(path):
            return handler_cls()
    return PlainFileHandler()


def build_file_metadata(file_path: Path, relative_path: Path) -> FileMetadata:
    handler = resolve_handler(file_path)
    result = handler.process(file_path)
    return FileMetadata(
        file_name=file_path.name,
        file_checksum_sha256=result.content_id,
        file_relative_path=relative_path,
        storage_unit=result.storage_unit,
        unit_manifest=result.manifest,
    )


def content_keys_for_metadata(file_metadata: FileMetadata) -> Set[str]:
    if file_metadata.storage_unit == SafetensorsHandler.kind and file_metadata.unit_manifest:
        keys: Set[str] = set()
        for part in file_metadata.unit_manifest.get("parts", []):
            keys.add(part["content_hash"])
        return keys
    return {file_metadata.file_checksum_sha256}


def content_keys_for_index(folder_index: Iterable[FileMetadata]) -> Set[str]:
    keys: Set[str] = set()
    for meta in folder_index:
        keys |= content_keys_for_metadata(meta)
    return keys
