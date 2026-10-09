from pathlib import Path
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from express.base.progress import TransferSession
from express.base.transfer import open_remote_file, pull_file, push_chunk


def test_push_chunk_with_byte_progress(tmp_path):
    path = tmp_path / "chunk.bin"
    path.write_bytes(b"data")
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    seen = []

    def on_bytes(n: int) -> None:
        seen.append(n)

    def upload(*_args, **kwargs):
        callback = kwargs.get("Callback")
        if callback is not None:
            callback(len(path.read_bytes()))

    remote.s3_client.upload_file.side_effect = upload
    push_chunk(remote, path, "key", byte_progress=on_bytes)
    assert seen == [4]


def test_push_chunk_with_session(tmp_path):
    path = tmp_path / "chunk.bin"
    path.write_bytes(b"data")
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    with TransferSession("Push", total_files=1) as session:
        push_chunk(remote, path, "key", session=session)


def test_pull_with_session_and_force_resync(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"old")
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.head_object.return_value = {"ContentLength": 3}

    def download(bucket, key, filename, Callback=None):
        Path(filename).write_bytes(b"new")

    remote.s3_client.download_file.side_effect = download
    with TransferSession("Pull", total_files=1) as session:
        pull_file(remote, Path("k"), path, "mismatch", force=True, session=session)
    assert path.read_bytes() == b"new"


def test_open_remote_file_index_branch(tmp_path):
    from express.base.model import MODEL_INDEX_FILE_NAME

    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.head_object.return_value = {"ContentLength": 2}
    local = tmp_path / MODEL_INDEX_FILE_NAME

    def download(bucket, key, filename, Callback=None):
        Path(filename).write_bytes(b"{}")

    remote.s3_client.download_file.side_effect = download
    with open_remote_file(remote, Path("k"), local, None, force=True) as handle:
        assert handle.read() == b"{}"


def test_open_remote_file(tmp_path):
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.head_object.return_value = {"ContentLength": 2}
    local = tmp_path / "remote.json"

    def download(bucket, key, filename, Callback=None):
        Path(filename).write_bytes(b"{}")

    remote.s3_client.download_file.side_effect = download
    with open_remote_file(remote, Path("k"), local, None, force=True) as handle:
        assert handle.read() == b"{}"
