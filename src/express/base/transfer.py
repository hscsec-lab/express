from contextlib import contextmanager
from io import BufferedReader
from pathlib import Path
from typing import Any, Generator, Optional

from rich.text import Text

from express import console
from express.base.client import is_remote_file_exists, Remote
from express.base.file import FileMetadata, fast_checksum
from express.base.model import Model, MODEL_INDEX_FILE_NAME
from express.base.progress import TransferSession, file_size


def push_chunk(
        remote: Remote,
        local_file_path: Path,
        remote_file_name: str = None,
        force: bool = False,
        session: Optional[TransferSession] = None,
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

    callback = None
    if session is not None:
        session.on_start(Path(local_file_path).name, file_size(local_file_path))
        callback = session.on_progress

    remote.s3_client.upload_file(
        str(local_file_path),
        remote.s3_bucket,
        remote_file_name,
        Callback=callback,
    )
    if session is not None:
        session.on_finish()


def push_chunk_bytes(
        remote: Remote,
        payload: bytes,
        remote_file_name: str,
        *,
        force: bool = False,
        session: Optional[TransferSession] = None,
) -> None:
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    try:
        push_chunk(remote, tmp_path, remote_file_name, force=force, session=session)
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
) -> None:
    """Uploads a local file to the remote S3 bucket if it doesn't already exist."""
    local_file_path = model.path / file_metadata.file_relative_path
    if file_metadata.file_name == MODEL_INDEX_FILE_NAME:
        remote_file_name = f"{model_metadata_torrent}.{MODEL_INDEX_FILE_NAME}"
        push_chunk(remote, local_file_path, remote_file_name, session=session)
        return
    from express.base.storage.transfer_ops import upload_storage_parts

    upload_storage_parts(remote, local_file_path, file_metadata, session=session)


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

    remote.s3_client.download_file(
        remote.s3_bucket,
        str(remote_file_path),
        str(local_file_path),
        Callback=callback,
    )
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
