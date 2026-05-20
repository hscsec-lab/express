from contextlib import contextmanager
from io import BufferedReader
from pathlib import Path
from typing import Any, Generator

from rich.text import Text

from express import console
from express.base.client import is_remote_file_exists, Remote
from express.base.file import FileMetadata, fast_checksum
from express.base.model import Model, MODEL_INDEX_FILE_NAME
from express.base.s3 import ProgressPercentage, DownloadProgressSimple


def push_chunk(remote: Remote, local_file_path: Path,remote_file_name: str = None,force: bool = False) -> None:
    """
    Push a chunk file to the remote server. The remote file name defaults to the SHA256
    checksum of the chunk. Skips upload if the file already exists.
    :param remote_file_name:
    :param remote:
    :param local_file_path:
    :param force:
    :return:
    """
    if not remote_file_name:
        remote_file_name = fast_checksum(local_file_path)

    if is_remote_file_exists(remote.s3_client, remote.s3_bucket, remote_file_name) and not force:
        console.print(Text.assemble("✓ Skip ", (remote_file_name, "dim"), ": already exists"))
        return

    remote.s3_client.upload_file(
        str(local_file_path),
        remote.s3_bucket,
        remote_file_name,
        Callback=ProgressPercentage(str(local_file_path))
    )


def push_file(model: Model, remote: Remote, file_metadata: FileMetadata, model_metadata_torrent: str) -> None:
    """Uploads a local file to the remote S3 bucket if it doesn't already exist."""
    remote_file_name = file_metadata.file_checksum_sha256
    if file_metadata.file_name == MODEL_INDEX_FILE_NAME:
        remote_file_name = f"{model_metadata_torrent}.{MODEL_INDEX_FILE_NAME}"
    local_file_path = model.path / file_metadata.file_relative_path

    push_chunk(remote, local_file_path, remote_file_name)


def pull_file(remote: Remote, remote_file_path: Path, local_file_path: Path,
              file_checksum_sha256: str | None, force: bool = False) -> Path:
    """Downloads a file from S3, with graceful handling of checksum algorithm transitions."""

    if local_file_path.exists():
        if file_checksum_sha256:
            # Use the new fast checksum algorithm
            current_local_hash = fast_checksum(local_file_path)

            if current_local_hash == file_checksum_sha256:
                console.print(Text.assemble("✓ Match ", (str(local_file_path), "dim"), " (fast-check)"))
                return local_file_path

            # Handling when Hash does not match
            print(f"force: {force}")
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
            return local_file_path

    # Execute download logic
    local_file_path.parent.mkdir(parents=True, exist_ok=True)
    remote.s3_client.download_file(
        remote.s3_bucket,
        str(remote_file_path),
        local_file_path,
        Callback=DownloadProgressSimple(str(local_file_path.name))
    )

    # After downloading, it is recommended to immediately verify with the new algorithm and store it in the local cache (if there will be a local metadata library in the future)
    return local_file_path
@contextmanager
def open_remote_file(remote: Remote, remote_file_path: Path, local_file_path: Path, file_checksum_sha256: str | None,
                     force: bool = False) -> Generator[BufferedReader, Any, None]:
    """Context manager that downloads a remote file and yields a read-only file object."""
    if local_file_path.name == MODEL_INDEX_FILE_NAME:
        # If it is an index file, then the remote file name is f"{model_metadata_torrent}.{MODEL_INDEX_FILE_NAME}"
        ...
    pull_file(remote, remote_file_path, local_file_path, file_checksum_sha256, force)
    f = None
    try:
        f = local_file_path.open("rb")
        yield f
    finally:
        if f:
            f.close()
