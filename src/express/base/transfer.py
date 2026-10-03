from contextlib import contextmanager
from io import BufferedReader
from pathlib import Path
from typing import Any, Callable, Generator, Optional

from rich.text import Text

from express import console
from express.base.client import is_remote_file_exists, Remote
from express.base.file import FileMetadata, fast_checksum
from express.base.model import Model, MODEL_INDEX_FILE_NAME
from express.base.progress import TransferSession, file_size
from express.base.storage.base import StorageBlob


def _transfer_cleanup(session: Optional[TransferSession]):
    return session.cleanup if session is not None else None


class _FileRangeReader:
    """File-like reader for upload_fileobj over a byte range."""

    def __init__(self, path: Path, offset: int, length: int):
        self._handle = path.open("rb")
        self._handle.seek(offset)
        self._remaining = length

    def read(self, amt: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if amt < 0 or amt > self._remaining:
            amt = self._remaining
        data = self._handle.read(amt)
        self._remaining -= len(data)
        return data

    def close(self) -> None:
        self._handle.close()


def push_storage_blob(
        remote: Remote,
        blob: StorageBlob,
        remote_file_name: str | None = None,
        *,
        force: bool = False,
        session: Optional[TransferSession] = None,
        label: str | None = None,
        aggregate_session: Optional[TransferSession] = None,
) -> None:
    remote_file_name = remote_file_name or blob.content_hash
    if is_remote_file_exists(remote.s3_client, remote.s3_bucket, remote_file_name) and not force:
        if aggregate_session is not None:
            aggregate_session.on_aggregate_blob_skip(blob.payload_length)
        elif session is None:
            console.print(Text.assemble("✓ Skip ", (remote_file_name[:16] + "…", "dim"), ": exists"))
        else:
            session.skip(advance_overall=False)
        return

    callback: Callable[[int], None] | None = None
    display = label or remote_file_name[:24]
    total = blob.payload_length
    if aggregate_session is not None:
        callback = aggregate_session.on_aggregate_progress
    elif session is not None:
        session.on_start(display, total)
        callback = session.on_progress

    cleanup = _transfer_cleanup(session)
    if blob.source_path is not None and blob.byte_length:
        if cleanup is not None:
            cleanup.track_remote_key(remote, remote_file_name)
        try:
            reader = _FileRangeReader(blob.source_path, blob.byte_offset, blob.byte_length)
            try:
                remote.s3_client.upload_fileobj(
                    reader,
                    remote.s3_bucket,
                    remote_file_name,
                    Callback=callback,
                )
            finally:
                reader.close()
        except BaseException:
            raise
        else:
            if cleanup is not None:
                cleanup.clear_remote_key(remote, remote_file_name)
    else:
        if aggregate_session is not None:
            push_chunk_bytes(
                remote,
                blob.read_payload(),
                remote_file_name,
                force=force,
                session=None,
                byte_progress=aggregate_session.on_aggregate_progress,
            )
            aggregate_session.on_blob_uploaded()
        else:
            push_chunk_bytes(remote, blob.read_payload(), remote_file_name, force=force, session=session)
        return

    if aggregate_session is not None:
        aggregate_session.on_blob_uploaded()
    elif session is not None:
        session.on_finish(advance_overall=False)


def push_chunk(
        remote: Remote,
        local_file_path: Path,
        remote_file_name: str = None,
        force: bool = False,
        session: Optional[TransferSession] = None,
        byte_progress: Callable[[int], None] | None = None,
) -> None:
    """
    Push a chunk file to the remote server. The remote file name defaults to the SHA256
    checksum of the chunk. Skips upload if the file already exists.
    """
    if not remote_file_name:
        remote_file_name = fast_checksum(local_file_path)

    if is_remote_file_exists(remote.s3_client, remote.s3_bucket, remote_file_name) and not force:
        if session is None:
            console.print(Text.assemble("✓ Skip ", (remote_file_name, "dim"), ": already exists"))
        else:
            session.skip()
        return

    callback: Callable[[int], None] | None = None
    if byte_progress is not None:
        callback = byte_progress
    elif session is not None:
        session.on_start(Path(local_file_path).name, file_size(local_file_path))
        callback = session.on_progress

    cleanup = _transfer_cleanup(session)
    if cleanup is not None:
        cleanup.track_remote_key(remote, remote_file_name)
    try:
        remote.s3_client.upload_file(
            str(local_file_path),
            remote.s3_bucket,
            remote_file_name,
            Callback=callback,
        )
    except BaseException:
        raise
    else:
        if cleanup is not None:
            cleanup.clear_remote_key(remote, remote_file_name)
    if session is not None and byte_progress is None:
        session.on_finish()


def push_chunk_bytes(
        remote: Remote,
        payload: bytes,
        remote_file_name: str,
        *,
        force: bool = False,
        session: Optional[TransferSession] = None,
        byte_progress: Callable[[int], None] | None = None,
) -> None:
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    try:
        push_chunk(
            remote,
            tmp_path,
            remote_file_name,
            force=force,
            session=session,
            byte_progress=byte_progress,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def pull_chunk_bytes(
        remote: Remote,
        remote_file_name: str,
        session: Optional[TransferSession] = None,
) -> bytes:
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pull_file(remote, Path(remote_file_name), tmp_path, None, force=True, session=session)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def push_file(
        model: Model,
        remote: Remote,
        file_metadata: FileMetadata,
        model_metadata_torrent: str,
        session: Optional[TransferSession] = None,
        blob_concurrency: int | None = None,
) -> None:
    """Uploads a local file to the remote S3 bucket if it doesn't already exist."""
    from express.base.storage.transfer_ops import DEFAULT_BLOB_CONCURRENCY, upload_storage_parts

    local_file_path = model.path / file_metadata.file_relative_path
    if file_metadata.file_name == MODEL_INDEX_FILE_NAME:
        remote_file_name = f"{model_metadata_torrent}.{MODEL_INDEX_FILE_NAME}"
        push_chunk(remote, local_file_path, remote_file_name, session=session)
        return
    workers = DEFAULT_BLOB_CONCURRENCY if blob_concurrency is None else blob_concurrency
    upload_storage_parts(
        remote,
        local_file_path,
        file_metadata,
        session=session,
        blob_concurrency=workers,
    )


def pull_file(
        remote: Remote,
        remote_file_path: Path,
        local_file_path: Path,
        file_checksum_sha256: str | None,
        force: bool = False,
        session: Optional[TransferSession] = None,
) -> Path:
    """Downloads a file from S3, with graceful handling of checksum algorithm transitions."""

    if local_file_path.exists():
        if file_checksum_sha256:
            current_local_hash = fast_checksum(local_file_path)

            if current_local_hash == file_checksum_sha256:
                if session is None:
                    console.print(Text.assemble("✓ Match ", (str(local_file_path), "dim"), " (fast-check)"))
                else:
                    session.skip()
                return local_file_path

            if force:
                console.print(Text.assemble("🔄 Re-syncing ", str(local_file_path), " due to hash update..."))
            else:
                console.print(Text.assemble(
                    "❓ Notice ", (f"{local_file_path.name}", "bold yellow"),
                    ": local hash mismatch (May mismatch express version). ",
                    ("Algorithm mismatch or partial file?", "italic dim")
                ))
                console.print(Text.assemble(
                    "   └─ ", ("Use --force to sync with Express-Checksum index.", "dim")
                ))
                if session is not None:
                    session.skip()
                return local_file_path

    local_file_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    try:
        head = remote.s3_client.head_object(Bucket=remote.s3_bucket, Key=str(remote_file_path))
        total_bytes = int(head.get("ContentLength") or 0)
    except Exception:
        total_bytes = 0

    callback = None
    if session is not None:
        session.on_start(local_file_path.name, total_bytes)
        callback = session.on_progress

    cleanup = _transfer_cleanup(session)
    if cleanup is not None:
        cleanup.track_local_file(local_file_path)
    try:
        remote.s3_client.download_file(
            remote.s3_bucket,
            str(remote_file_path),
            str(local_file_path),
            Callback=callback,
        )
    except BaseException:
        raise
    else:
        if cleanup is not None:
            cleanup.clear_local_file(local_file_path)
    if session is not None:
        session.on_finish()
    return local_file_path


@contextmanager
def open_remote_file(
        remote: Remote,
        remote_file_path: Path,
        local_file_path: Path,
        file_checksum_sha256: str | None,
        force: bool = False,
) -> Generator[BufferedReader, Any, None]:
    """Context manager that downloads a remote file and yields a read-only file object."""
    if local_file_path.name == MODEL_INDEX_FILE_NAME:
        ...
    pull_file(remote, remote_file_path, local_file_path, file_checksum_sha256, force)
    f = None
    try:
        f = local_file_path.open("rb")
        yield f
    finally:
        if f:
            f.close()
