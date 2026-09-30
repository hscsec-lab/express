from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from express.base.client import Remote
from express.base.file import FileMetadata
from express.base.progress import TransferSession
from express.base.storage.registry import resolve_handler
from express.base.storage.safetensors import SafetensorsHandler
from express.base.transfer import push_chunk, pull_chunk_bytes


def upload_storage_parts(
        remote: Remote,
        local_file_path: Path,
        file_metadata: FileMetadata,
        *,
        force: bool = False,
        session: Optional[TransferSession] = None,
) -> None:
    handler = resolve_handler(local_file_path)
    result = handler.process(local_file_path)
    seen: set[str] = set()
    for blob in result.iter_blobs():
        if blob.content_hash in seen:
            continue
        seen.add(blob.content_hash)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(blob.data)
            tmp_path = Path(tmp.name)
        try:
            push_chunk(remote, tmp_path, blob.content_hash, force=force, session=session)
        finally:
            tmp_path.unlink(missing_ok=True)


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
        blobs[content_hash] = pull_chunk_bytes(remote, content_hash, session=session)
    handler.restore(manifest, blobs, local_file_path)
    if not handler.verify_local(local_file_path, file_metadata.file_checksum_sha256, manifest):
        raise RuntimeError(f"restored safetensors failed verification: {local_file_path}")
    return local_file_path
