from contextlib import contextmanager
from io import BufferedReader
from pathlib import Path
from typing import Any, Generator

from rich.text import Text
from simple_file_checksum import get_checksum

from src import console
from base.client import is_remote_file_exists, Remote
from base.file import FileMetadata
from base.model import Model, MODEL_INDEX_FILE_NAME
from base.s3 import ProgressPercentage, DownloadProgressSimple


def push_chunk(remote: Remote, local_file_path: Path,remote_file_name: str = None,force: bool = False) -> None:
    """
    推送一个chunk文件到远程服务器，文件名为chunk的sha256值，如果文件已经存在则跳过。
    :param remote_file_name:
    :param remote:
    :param local_file_path:
    :param force:
    :return:
    """
    if not remote_file_name:
        remote_file_name = get_checksum(local_file_path)

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


def pull_file(remote: Remote, remote_file_path: Path, local_file_path: Path, file_checksum_sha256: str | None,
              force: bool = False) -> Path:
    """Downloads a file from S3, skipping or aborting based on local checksum validation."""
    if local_file_path.exists():
        if file_checksum_sha256:
            local_checksum = get_checksum(local_file_path)
            if local_checksum == file_checksum_sha256:
                console.print(Text.assemble("✓ Skip ", (str(local_file_path), "dim"), ": already exists"))
                return local_file_path
            else:
                if force:
                    console.print(Text.assemble("⚠️  Overwriting ", str(local_file_path), ": checksum mismatch"))
                else:
                    console.print(Text.assemble("✗ Skip ", str(local_file_path),
                                                ": checksum mismatch (use --force to overwrite)"))
                    return local_file_path

    local_file_path.parent.mkdir(parents=True, exist_ok=True)
    remote.s3_client.download_file(
        remote.s3_bucket,
        str(remote_file_path),
        local_file_path,
        Callback=DownloadProgressSimple(str(local_file_path.name))
    )
    return local_file_path

@contextmanager
def open_remote_file(remote: Remote, remote_file_path: Path, local_file_path: Path, file_checksum_sha256: str | None,
                     force: bool = False) -> Generator[BufferedReader, Any, None]:
    """Context manager that downloads a remote file and yields a read-only file object."""
    if local_file_path.name == MODEL_INDEX_FILE_NAME:
        # 如果是索引文件，那么远程文件名是f"{model_metadata_torrent}.{MODEL_INDEX_FILE_NAME}"
        ...
    pull_file(remote, remote_file_path, local_file_path, file_checksum_sha256, force)
    f = None
    try:
        f = local_file_path.open("rb")
        yield f
    finally:
        if f:
            f.close()
