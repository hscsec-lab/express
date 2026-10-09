import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from express.base.data import Torrent, get_torrent
from express.base.model import Metadata, MODEL_INDEX_FILE_NAME
from express.base.remote import (
    _get_referenced_chunks,
    _load_remote_index,
    _print_entries,
    delete,
    ls,
    pull,
    pull_index_with_torrent,
    pull_model,
    RemoteEntry,
)


def _metadata(name="demo"):
    return Metadata(name=name, version="0.1.0", authors="a", tags=["t"])


def test_load_remote_index_success():
    remote = MagicMock()
    remote.s3_bucket = "b"
    index = {"folder_index": []}
    remote.s3_client.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps(index).encode("utf-8"))
    }
    loaded = _load_remote_index(remote, Torrent("abc"))
    assert loaded.folder_index == []


def test_load_remote_index_invalid_json():
    remote = MagicMock()
    remote.s3_client.get_object.return_value = {
        "Body": MagicMock(read=lambda: b"not-json")
    }
    with pytest.raises(RuntimeError, match="Invalid remote index"):
        _load_remote_index(remote, Torrent("abc"))


def test_get_referenced_chunks_aborts_on_bad_index():
    remote = MagicMock()
    page_iterator = MagicMock()
    page_iterator.search.return_value = iter(["bad.express-index.json"])
    paginator = MagicMock()
    paginator.paginate.return_value = page_iterator
    remote.s3_client.get_paginator.return_value = paginator
    remote.s3_client.get_object.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="Aborting delete"):
        _get_referenced_chunks(remote)


def test_load_remote_index_not_found():
    remote = MagicMock()
    remote.s3_client.get_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "GetObject"
    )
    remote.s3_client.exceptions.ClientError = ClientError
    with pytest.raises(ClientError):
        _load_remote_index(remote, Torrent("abc"))


def test_delete_dry_run():
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.return_value = {}
    meta = _metadata()
    torrent = get_torrent(meta)
    index_key = f"{torrent}.{MODEL_INDEX_FILE_NAME}"
    index = {"folder_index": []}
    remote.s3_client.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps(index).encode("utf-8"))
    }
    page_iterator = MagicMock()
    page_iterator.search.return_value = iter([])
    paginator = MagicMock()
    paginator.paginate.side_effect = [page_iterator, [{"Contents": []}]]
    remote.s3_client.get_paginator.return_value = paginator
    delete(torrent, remote, yes=False, dry_run=True)


def test_print_entries_and_ls(capsys):
    first = _metadata("one")
    second = _metadata("two")
    second.emails = "e@x.com"
    entries = [
        RemoteEntry(torrent=Torrent("t1"), metadata=first),
        RemoteEntry(torrent=Torrent("t2"), metadata=second),
    ]
    _print_entries(entries)
    out = capsys.readouterr().out
    assert "one" in out
    remote = MagicMock()
    with patch("express.base.remote.catalog", return_value=entries):
        ls(remote, torrent=False)
        ls(remote, torrent=True)


def test_delete_reports_missing_owned_keys():
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.return_value = {}
    meta = _metadata()
    torrent = get_torrent(meta)
    index = {
        "folder_index": [
            {
                "file_name": "a.bin",
                "file_checksum_sha256": "gone",
                "file_relative_path": "a.bin",
            }
        ]
    }
    remote.s3_client.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps(index).encode("utf-8"))
    }
    page_iterator = MagicMock()
    page_iterator.search.return_value = iter([])
    paginator = MagicMock()
    paginator.paginate.side_effect = [page_iterator, [{"Contents": []}]]
    remote.s3_client.get_paginator.return_value = paginator
    delete(torrent, remote, yes=False, dry_run=True)


def test_pull_index_with_torrent(tmp_path):
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.return_value = {}
    payload = {"folder_index": []}

    def download(bucket, key, filename, Callback=None):
        __import__("pathlib").Path(filename).write_bytes(json.dumps(payload).encode("utf-8"))

    remote.s3_client.download_file.side_effect = download
    index_bytes = json.dumps(payload).encode("utf-8")
    remote.s3_client.head_object.return_value = {"ContentLength": len(index_bytes)}
    data = pull_index_with_torrent(Torrent("abc"), remote)
    assert data == payload


def test_get_referenced_chunks_skips_exclude():
    remote = MagicMock()
    remote.s3_bucket = "b"
    index_key = "abc.express-index.json"
    remote.s3_client.get_paginator.return_value.paginate.return_value.search.return_value = iter([index_key])
    index = {"folder_index": []}
    remote.s3_client.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps(index).encode("utf-8"))
    }
    refs = _get_referenced_chunks(remote, exclude_torrent=Torrent("abc"))
    assert refs == set()


@patch("express.base.remote.pull_model_with_index")
@patch("express.base.remote.pull_index_with_torrent")
@patch("express.base.remote.Model")
def test_pull_model_and_pull(mock_model, mock_index, mock_pull_index, tmp_path):
    mock_index.return_value = {"folder_index": []}
    mock_pull_index.return_value = MagicMock()
    mock_model.return_value = MagicMock()
    remote = MagicMock()
    meta = _metadata()
    torrent = get_torrent(meta)
    pull_model(torrent, remote, mock_model.return_value, force=False)
    with patch("express.base.remote.from_torrent", return_value=meta):
        with patch("express.base.remote.local_workdir", return_value=tmp_path):
            pull(torrent, remote, force=False)
