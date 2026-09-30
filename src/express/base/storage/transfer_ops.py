from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

from express.base.client import Remote
from express.base.file import FileMetadata
from express.base.progress import TransferSession
from express.base.storage.base import StorageBlob
from express.base.storage.registry import resolve_handler
from express.base.storage.safetensors import SafetensorsHandler
from express.base.transfer import push_storage_blob


def _blobs_from_manifest_file(path: Path, manifest: dict) -> list[StorageBlob]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        data_base = 8 + header_size
    header = manifest["header"]
    blobs: list[StorageBlob] = []
    for part in manifest["parts"]:
        name = part["name"]
        start, end = header[name]["data_offsets"]
        blobs.append(
            StorageBlob(
                content_hash=part["content_hash"],
                source_path=path,
                byte_offset=data_base + start,
                byte_length=part["byte_length"],
            )
        )
    return blobs


def upload_storage_parts(
        remote: Remote,
        local_file_path: Path,
        file_metadata: FileMetadata,
        *,
        force: bool = False,
        session: Optional[TransferSession] = None,
) -> None:
    handler = resolve_handler(local_file_path)
    manifest = file_metadata.unit_manifest

    if (
        file_metadata.storage_unit == SafetensorsHandler.kind
        and manifest
        and isinstance(handler, SafetensorsHandler)
        and handler.verify_local(local_file_path, file_metadata.file_checksum_sha256, manifest)
    ):
        blobs = _blobs_from_manifest_file(local_file_path, manifest)
    else:
        result = handler.process(local_file_path)
        blobs = list(result.blobs)

    seen: set[str] = set()
    for blob in blobs:
        if blob.content_hash in seen:
            continue
        seen.add(blob.content_hash)
        push_storage_blob(
            remote,
            blob,
            blob.content_hash,
            force=force,
            session=session,
            label=local_file_path.name,
        )


def materialize_local_file(
        remote: Remote,
        file_metadata: FileMetadata,
        local_file_path: Path,
        *,
        force: bool = False,
        session: Optional[TransferSession] = None,
) -> Path:
    if (
        file_metadata.storage_unit == SafetensorsHandler.kind
        and file_metadata.unit_manifest
    ):
        return _materialize_safetensors(remote, file_metadata, local_file_path, force=force, session=session)
    return _materialize_plain(remote, file_metadata, local_file_path, force=force, session=session)


def _materialize_plain(
        remote: Remote,
        file_metadata: FileMetadata,
        local_file_path: Path,
        *,
        force: bool = False,
        session: Optional[TransferSession] = None,
) -> Path:
    from express.base.transfer import pull_file

    return pull_file(
        remote,
        Path(file_metadata.file_checksum_sha256),
        local_file_path,
        file_metadata.file_checksum_sha256,
        force=force,
        session=session,
    )


def _materialize_safetensors(
        remote: Remote,
        file_metadata: FileMetadata,
        local_file_path: Path,
        *,
        force: bool = False,
        session: Optional[TransferSession] = None,
) -> Path:
    handler = SafetensorsHandler()
    manifest = file_metadata.unit_manifest or {}
    if local_file_path.exists() and not force:
        if handler.verify_local(local_file_path, file_metadata.file_checksum_sha256, manifest):
            if session is not None:
                session.skip()
            return local_file_path

    blobs: dict[str, bytes] = {}
    for part in manifest.get("parts", []):
        content_hash = part["content_hash"]
        from express.base.transfer import pull_chunk_bytes

        blobs[content_hash] = pull_chunk_bytes(remote, content_hash, session=session)

    cleanup = session.cleanup if session is not None else None
    if cleanup is not None:
        cleanup.track_local_file(local_file_path)
    try:
        handler.restore(manifest, blobs, local_file_path)
        if not handler.verify_local(local_file_path, file_metadata.file_checksum_sha256, manifest):
            raise RuntimeError(f"restored safetensors failed verification: {local_file_path}")
    except BaseException:
        raise
    else:
        if cleanup is not None:
            cleanup.clear_local_file(local_file_path)
    return local_file_path
