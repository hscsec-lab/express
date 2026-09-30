from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from express.base.data import Torrent
from express.base.file import FileMetadata
from express.base.model import MODEL_INDEX_FILE_NAME
from express.base.remote import (
    _format_bytes,
    _get_index_remote_key,
    _get_torrent_from_index_key,
    bucket_stats,
    catalog,
    du,
    local_workdir,
    pull_model_with_index,
    push,
    search,
)


def test_index_key_helpers():
    torrent = Torrent("abc123")
    key = _get_index_remote_key(torrent)
    assert key.endswith(MODEL_INDEX_FILE_NAME)
    assert _get_torrent_from_index_key(key) == torrent
    with pytest.raises(ValueError):
        _get_torrent_from_index_key("not-an-index")


def test_format_bytes():
    assert "KB" in _format_bytes(2048)
    assert "PB" in _format_bytes(1024**5)


def test_local_workdir(monkeypatch):
    monkeypatch.setenv("LOCAL_WORKDIR", "/tmp/models")
    assert local_workdir() == __import__("pathlib").Path("/tmp/models")


def test_pull_model_with_index_plain(tmp_path):
    payload = tmp_path / "chunk"
    payload.write_bytes(b"data")
    from express.base.file import fast_checksum

    meta = FileMetadata(
        file_name="chunk",
        file_checksum_sha256=fast_checksum(payload),
        file_relative_path=__import__("pathlib").Path("chunk"),
    )
    index = {"folder_index": [meta.model_dump(mode="json")]}
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.return_value = {"ContentLength": 4}

    def download(bucket, key, filename, Callback=None):
        __import__("pathlib").Path(filename).write_bytes(b"data")

    remote.s3_client.download_file.side_effect = download
    out = tmp_path / "model"
    out.mkdir()
    pull_model_with_index(index, remote, out, force=True)
    assert (out / "chunk").read_bytes() == b"data"


def test_push_model(tmp_path, monkeypatch):
    from express.base.model import Model

    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "a.txt").write_text("x", encoding="utf-8")
    model = Model(model_dir)
    model.create_metadata(name="m", version="0.1.0")
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    push(model, remote)
    assert remote.s3_client.upload_fileobj.called or remote.s3_client.upload_file.called


def test_catalog_and_search_empty():
    remote = MagicMock()
    remote.s3_client.get_paginator.return_value.paginate.return_value.search.return_value = iter([])
    assert catalog(remote) == []
    search(remote, query="none")


def test_bucket_stats_and_du(capsys):
    remote = MagicMock()
    remote.s3_client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "a", "Size": 10}, {"Key": "b", "Size": 5}]}
    ]
    assert bucket_stats(remote) == (2, 15)
    du(remote)
    captured = capsys.readouterr().out
    assert "Objects: 2" in captured
