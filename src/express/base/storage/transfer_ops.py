from __future__ import annotations

import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from express.base.client import Remote
from express.base.file import FileMetadata
from express.base.progress import TransferSession
from express.base.storage.base import StorageBlob
from express.base.storage.registry import resolve_handler
from express.base.storage.safetensors import SafetensorsHandler
from express.base.transfer import push_storage_blob

DEFAULT_BLOB_CONCURRENCY = 5


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


def _upload_blob_worker(
        remote: Remote,
        blob: StorageBlob,
        *,
        force: bool,
        session: Optional[TransferSession],
        label: str,
        aggregate_session: Optional[TransferSession],
) -> None:
    push_storage_blob(
        remote,
        blob,
        blob.content_hash,
        force=force,
        session=session,
        label=label,
        aggregate_session=aggregate_session,
    )


def upload_storage_parts(
        remote: Remote,
        local_file_path: Path,
        file_metadata: FileMetadata,
        *,
        force: bool = False,
        session: Optional[TransferSession] = None,
        blob_concurrency: int = DEFAULT_BLOB_CONCURRENCY,
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

    unique_blobs: list[StorageBlob] = []
    seen: set[str] = set()
    for blob in blobs:
        if blob.content_hash in seen:
            continue
        seen.add(blob.content_hash)
        unique_blobs.append(blob)

    label = local_file_path.name
    workers = max(1, int(blob_concurrency))
    use_aggregate = session is not None and len(unique_blobs) > 1
    if use_aggregate:
        total_bytes = sum(blob.payload_length for blob in unique_blobs)
        session.begin_blob_batch(label, len(unique_blobs), total_bytes)
    aggregate = session if use_aggregate else None

    try:
        if workers == 1 or len(unique_blobs) <= 1:
            for blob in unique_blobs:
                _upload_blob_worker(
                    remote,
                    blob,
                    force=force,
                    session=session,
                    label=label,
                    aggregate_session=aggregate,
                )
        else:
            max_workers = min(workers, len(unique_blobs))
            pool = ThreadPoolExecutor(max_workers=max_workers)
            futures = [
                pool.submit(
                    _upload_blob_worker,
                    remote,
                    blob,
                    force=force,
                    session=session,
                    label=label,
                    aggregate_session=aggregate,
                )
                for blob in unique_blobs
            ]
            try:
                for future in as_completed(futures):
                    future.result()
            except KeyboardInterrupt:
                for future in futures:
                    future.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
    finally:
        if use_aggregate and session is not None:
            session.finish_blob_batch()

    if session is not None:
        session.complete_file()


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
