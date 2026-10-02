from pathlib import Path
from unittest.mock import MagicMock

import pytest

from express.base.file import (
    FileMetadata,
    fast_checksum,
    fast_range_digest,
    generate_index,
    get_remote_chunk_metadata_from_index,
)
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


def test_generate_index_safetensors_shows_progress(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    save_file({"w": np.array([1.0, 2.0], dtype=np.float32)}, tmp_path / "w.safetensors")
    idx = generate_index(tmp_path)
    assert idx.folder_index[0].storage_unit == "safetensors_v1"


def test_generate_index_uses_storage_layer(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"invalid")
    idx = generate_index(tmp_path)
    assert len(idx.folder_index) == 1
    assert idx.folder_index[0].storage_unit == "plain"


def test_file_range_reader_exhausted(tmp_path):
    from express.base.transfer import _FileRangeReader

    path = tmp_path / "f.bin"
    path.write_bytes(b"ab")
    reader = _FileRangeReader(path, 0, 2)
    assert reader.read(1) == b"a"
    assert reader.read(10) == b"b"
    assert reader.read() == b""
    reader.close()


def test_push_storage_blob_from_file_range(tmp_path):
    from botocore.exceptions import ClientError

    from express.base.storage.base import StorageBlob, sha256_file_range
    from express.base.transfer import push_storage_blob

    path = tmp_path / "chunk.bin"
    payload = b"range-payload-data"
    path.write_bytes(payload)
    assert sha256_file_range(path, 0, len(payload)) == sha256_file_range(path, 0, len(payload))
    blob = StorageBlob(content_hash="key", source_path=path, byte_offset=0, byte_length=len(payload))
    remote = MagicMock()
    remote.s3_bucket = "bucket"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )

    def capture_fileobj(fileobj, bucket, key, Callback=None):
        assert fileobj.read() == payload

    remote.s3_client.upload_fileobj.side_effect = capture_fileobj
    push_storage_blob(remote, blob, "key")


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


def test_fast_range_digest_matches_small_range(tmp_path):
    import mmap

    payload = b"tensor-bytes-for-digest"
    path = tmp_path / "t.bin"
    path.write_bytes(payload)
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            assert fast_range_digest(mm, 0, len(payload)) == fast_range_digest(mm, 0, len(payload))


def test_fast_range_digest_empty_and_large_sampled(tmp_path):
    import mmap

    assert fast_range_digest(memoryview(b""), 0, 0) == fast_range_digest(memoryview(b""), 0, 0)
    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * (70000 * 4))
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            h1 = fast_range_digest(mm, 0, len(mm))
            h2 = fast_range_digest(mm, 0, len(mm))
    assert h1 == h2


def test_generate_index_throttles_tensor_callback(tmp_path, monkeypatch):
    import numpy as np
    from safetensors.numpy import save_file

    save_file(
        {"a": np.zeros(4, dtype=np.float32), "b": np.ones(4, dtype=np.float32)},
        tmp_path / "m.safetensors",
    )
    times = iter([0.0, 0.05, 10.0])

    monkeypatch.setattr("time.monotonic", lambda: next(times, 100.0))
    generate_index(tmp_path)
