from pathlib import Path
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from express.base.transfer import pull_file, push_chunk


def test_pull_skip_on_hash_match(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"same")
    from express.base.file import fast_checksum

    digest = fast_checksum(path)
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    out = pull_file(remote, Path(digest), path, digest, force=False)
    assert out == path
    remote.s3_client.download_file.assert_not_called()


def test_pull_mismatch_without_force(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"local")
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    out = pull_file(remote, Path("remote"), path, "other-hash", force=False)
    assert out == path
    remote.s3_client.download_file.assert_not_called()


def test_pull_downloads_when_missing(tmp_path):
    path = tmp_path / "nested" / "f.bin"
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.head_object.return_value = {"ContentLength": 3}

    def download(bucket, key, filename, Callback=None):
        Path(filename).write_bytes(b"new")

    remote.s3_client.download_file.side_effect = download
    pull_file(remote, Path("key"), path, None, force=True)
    assert path.read_bytes() == b"new"


def test_pull_head_object_failure_still_downloads(tmp_path):
    path = tmp_path / "f.bin"
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.head_object.side_effect = RuntimeError("no head")

    def download(bucket, key, filename, Callback=None):
        Path(filename).write_bytes(b"ok")

    remote.s3_client.download_file.side_effect = download
    pull_file(remote, Path("k"), path, None, force=True)
    assert path.read_bytes() == b"ok"


def test_push_chunk_default_remote_name(tmp_path):
    path = tmp_path / "chunk.bin"
    path.write_bytes(b"x")
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    push_chunk(remote, path)
    assert remote.s3_client.upload_file.called


def test_push_chunk_skip_existing():
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.return_value = {}
    path = Path(__file__)
    push_chunk(remote, path, "exists", force=False)
    remote.s3_client.upload_file.assert_not_called()
