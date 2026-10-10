from __future__ import annotations

import struct
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

from express.base.client import Remote
from express.base.file import FileMetadata
from express.base.progress import TransferSession
from express.base.storage.base import StorageBlob
from express.base.storage.registry import resolve_handler
from express.base.storage.safetensors import SafetensorsHandler
from express.base.transfer import push_storage_blob

DEFAULT_BLOB_CONCURRENCY = 5


def _blobs_from_manifest_file(path: Path, manifest: dict) -> list[tuple[str, StorageBlob]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        data_base = 8 + header_size
    header = manifest["header"]
    named: list[tuple[str, StorageBlob]] = []
    for part in manifest["parts"]:
        name = part["name"]
        start, end = header[name]["data_offsets"]
        named.append(
            (
                name,
                StorageBlob(
                    content_hash=part["content_hash"],
                    source_path=path,
                    byte_offset=data_base + start,
                    byte_length=part["byte_length"],
                ),
            )
        )
    return named


def _unique_named_blobs(named: list[tuple[str, StorageBlob]]) -> list[tuple[str, StorageBlob]]:
    unique: list[tuple[str, StorageBlob]] = []
    seen: set[str] = set()
    for tensor_name, blob in named:
        if blob.content_hash in seen:
            continue
        seen.add(blob.content_hash)
        unique.append((tensor_name, blob))
    return unique


class _UploadSlotPool:
    """Maps each in-flight tensor upload to one visible progress slot."""

    def __init__(self, session: TransferSession, slot_count: int) -> None:
        self._session = session
        self._slot_count = max(int(slot_count), 1)
        self._free: list[int] = list(range(self._slot_count))
        self._cond = threading.Condition()

    @contextmanager
    def occupy(self, tensor_name: str, total_bytes: int) -> Iterator[int]:
        with self._cond:
            while not self._free:
                self._cond.wait()
            slot = self._free.pop()
        self._session.on_slot_start(slot, tensor_name, total_bytes)
        try:
            yield slot
        finally:
            self._session.on_slot_done(slot)
            with self._cond:
                self._free.append(slot)
                self._cond.notify()


def _upload_blob_worker(
        remote: Remote,
        blob: StorageBlob,
        *,
        tensor_name: str,
        force: bool,
        session: Optional[TransferSession],
        decomposed_session: Optional[TransferSession],
        slot_pool: Optional[_UploadSlotPool],
) -> None:
    if decomposed_session is not None and slot_pool is not None:
        with slot_pool.occupy(tensor_name, blob.payload_length) as slot:
            push_storage_blob(
                remote,
                blob,
                blob.content_hash,
                force=force,
                session=session,
                decomposed_session=decomposed_session,
                slot=slot,
            )
        return

    push_storage_blob(
        remote,
        blob,
        blob.content_hash,
        force=force,
        session=session,
        label=tensor_name,
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
        named_blobs = _blobs_from_manifest_file(local_file_path, manifest)
    else:
        result = handler.process(local_file_path)
        parts = (result.manifest or {}).get("parts", [])
        if parts and len(parts) == len(result.blobs):
            named_blobs = list(zip([p["name"] for p in parts], result.blobs))
        else:
            named_blobs = [(local_file_path.name, blob) for blob in result.blobs]

    unique_named = _unique_named_blobs(named_blobs)
    workers = max(1, int(blob_concurrency))
    use_decomposed = (
        session is not None
        and file_metadata.storage_unit == SafetensorsHandler.kind
        and len(unique_named) >= 1
    )

    if use_decomposed:
        total_bytes = sum(blob.payload_length for _, blob in unique_named)
        session.begin_decomposed_file(
            local_file_path.name,
            total_bytes,
            slot_count=workers,
        )
    slot_pool = _UploadSlotPool(session, workers) if use_decomposed else None
    decomposed = session if use_decomposed else None

    try:
        if workers == 1 or len(unique_named) <= 1:
            for tensor_name, blob in unique_named:
                _upload_blob_worker(
                    remote,
                    blob,
                    tensor_name=tensor_name,
                    force=force,
                    session=session,
                    decomposed_session=decomposed,
                    slot_pool=slot_pool,
                )
        else:
            max_workers = min(workers, len(unique_named))
            pool = ThreadPoolExecutor(max_workers=max_workers)
            futures = [
                pool.submit(
                    _upload_blob_worker,
                    remote,
                    blob,
                    tensor_name=tensor_name,
                    force=force,
                    session=session,
                    decomposed_session=decomposed,
                    slot_pool=slot_pool,
                )
                for tensor_name, blob in unique_named
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
        if use_decomposed and session is not None:
            session.finish_decomposed_file()

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


def _unique_manifest_parts(parts: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen: set[str] = set()
    for part in parts:
        content_hash = part["content_hash"]
        if content_hash in seen:
            continue
        seen.add(content_hash)
        unique.append(part)
    return unique


def _download_manifest_part(
        remote: Remote,
        part: dict,
        *,
        byte_progress: Callable[[int], None] | None = None,
) -> tuple[str, bytes]:
    from express.base.transfer import pull_chunk_bytes

    content_hash = part["content_hash"]
    length = part.get("byte_length")
    known_size = int(length) if length is not None else None
    payload = pull_chunk_bytes(
        remote,
        content_hash,
        byte_progress=byte_progress,
        known_size=known_size,
    )
    return content_hash, payload


def _materialize_safetensors(
        remote: Remote,
        file_metadata: FileMetadata,
        local_file_path: Path,
        *,
        force: bool = False,
        session: Optional[TransferSession] = None,
        blob_concurrency: int = DEFAULT_BLOB_CONCURRENCY,
) -> Path:
    handler = SafetensorsHandler()
    manifest = file_metadata.unit_manifest or {}
    if local_file_path.exists() and not force:
        if handler.verify_local(local_file_path, file_metadata.file_checksum_sha256, manifest):
            if session is not None:
                session.skip()
            return local_file_path

    parts = _unique_manifest_parts(list(manifest.get("parts") or []))
    workers = max(1, int(blob_concurrency))
    use_decomposed = session is not None and len(parts) >= 1
    slot_pool = _UploadSlotPool(session, workers) if use_decomposed and session is not None else None
    if use_decomposed and session is not None:
        total_bytes = sum(int(part.get("byte_length") or 0) for part in parts)
        session.begin_decomposed_file(
            local_file_path.name,
            total_bytes,
            slot_count=min(workers, max(len(parts), 1)),
        )

    blobs: dict[str, bytes] = {}
    blobs_lock = threading.Lock()

    def fetch(part: dict) -> None:
        name = str(part.get("name") or part["content_hash"][:12])
        length = int(part.get("byte_length") or 0)
        if slot_pool is not None and session is not None:
            with slot_pool.occupy(name, length) as slot:
                content_hash, payload = _download_manifest_part(
                    remote,
                    part,
                    byte_progress=lambda n, s=slot: session.on_slot_progress(s, n),
                )
        else:
            content_hash, payload = _download_manifest_part(remote, part)
        with blobs_lock:
            blobs[content_hash] = payload

    try:
        if workers == 1 or len(parts) <= 1:
            for part in parts:
                fetch(part)
        else:
            max_workers = min(workers, len(parts))
            pool = ThreadPoolExecutor(max_workers=max_workers)
            futures = [pool.submit(fetch, part) for part in parts]
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
    finally:
        if use_decomposed and session is not None:
            session.finish_decomposed_file()

    if session is not None:
        session.complete_file()
    return local_file_path
