from pathlib import Path
from unittest.mock import MagicMock

import pytest

from express.base.file import FileMetadata, fast_checksum, generate_index, get_remote_chunk_metadata_from_index
from express.base.transfer import pull_chunk_bytes, push_chunk_bytes


def test_file_metadata_path_serialization():
    meta = FileMetadata(
        file_name="a.txt",
        file_checksum_sha256="abc",
        file_relative_path=Path("sub/a.txt"),
    )
    dumped = meta.model_dump()
    assert dumped["file_relative_path"] == "sub/a.txt"
    restored = FileMetadata.model_validate(dumped)
    assert restored.file_relative_path == Path("sub/a.txt")


def test_file_metadata_invalid_path():
    with pytest.raises(ValueError):
        FileMetadata(
            file_name="a",
            file_checksum_sha256="x",
            file_relative_path=123,  # type: ignore[arg-type]
        )


def test_get_remote_chunk_metadata_from_index():
    meta = FileMetadata(file_name="w.bin", file_checksum_sha256="h", file_relative_path=Path("w.bin"))
    index = __import__("express.base.file", fromlist=["FolderIndex"]).FolderIndex(folder_index=[meta])
    assert get_remote_chunk_metadata_from_index(index, "w.bin") == meta
    assert get_remote_chunk_metadata_from_index(index, "missing") is None


def test_generate_index_uses_storage_layer(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"invalid")
    idx = generate_index(tmp_path)
    assert len(idx.folder_index) == 1
    assert idx.folder_index[0].storage_unit == "plain"


def test_push_and_pull_chunk_bytes(tmp_path):
    remote = MagicMock()
    remote.s3_bucket = "bucket"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = Exception
    remote.s3_client.head_object.return_value = {}
    payload = b"hello-chunk"

    def capture_upload(local, bucket, key, Callback=None):
        assert Path(local).read_bytes() == payload

    remote.s3_client.upload_file.side_effect = capture_upload

    push_chunk_bytes(remote, payload, "hash-key")

    stored = tmp_path / "blob"
    stored.write_bytes(payload)

    def fake_download(bucket, key, filename, Callback=None):
        Path(filename).write_bytes(stored.read_bytes())

    remote.s3_client.head_object.return_value = {"ContentLength": len(payload)}
    remote.s3_client.download_file.side_effect = fake_download
    assert pull_chunk_bytes(remote, "hash-key") == payload


def test_get_remote_chunk_name():
    meta = FileMetadata(file_name="a", file_checksum_sha256="deadbeef", file_relative_path=Path("a"))
    assert meta.get_remote_chunk_name() == "deadbeef"


def test_fast_checksum_large_file_sampling(tmp_path):
    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * (70000 * 4))
    h1 = fast_checksum(path)
    h2 = fast_checksum(path)
    assert h1 == h2
