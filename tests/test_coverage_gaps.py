import json
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from pydantic import BaseModel

from express.base.client import Remote
from express.base.data import assert_serializable, search_models
from express.base.progress import TransferSession
from express.base.remote import _load_remote_index, catalog, delete
from express.base.storage.base import StorageBlob
from express.base.storage.safetensors import SafetensorsHandler, _compose_safetensors, _read_safetensors_layout
from express.base.storage.transfer_ops import _materialize_safetensors
from express.base.transfer import pull_file, push_chunk


class Item(BaseModel):
    name: str
    authors: str = ""
    emails: str = ""
    version: str = ""
    tags: list[str] = []


def test_assert_serializable_success():
    assert assert_serializable(Item(name="x")) is True


def test_search_models_email_and_version_filters():
    items = [Item(name="hello", emails="a@b.c", version="1.0.0", authors="Alice")]
    assert search_models(items, emails="a@b", version="1.0", author="ali")
    assert not search_models(items, version="9.9.9")
    assert search_models(items, query="hello")
    assert not search_models([Item(name="x", tags=["gpu"])], tag="cpu")
    assert not search_models(items, author="missing")
    assert not search_models(items, emails="zzz")


def test_storage_blob_byte_length():
    blob = StorageBlob(content_hash="h", data=b"1234")
    assert blob.byte_length == 4


def test_verify_local_manifest_mismatch(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"x")
    handler = __import__("express.base.storage.plain", fromlist=["PlainFileHandler"]).PlainFileHandler()
    result = handler.process(path)
    assert handler.verify_local(path, result.content_id, {"different": True}) is False


@patch.dict("os.environ", {"S3_AK": "a", "S3_SK": "s", "S3_ENDPOINT": "e"}, clear=True)
@patch("express.base.config.ensure_remote_config")
@patch("boto3.client")
@patch("boto3.resource")
def test_remote_missing_bucket_env(_res, _cli, _ensure):
    with pytest.raises(EnvironmentError, match="S3_BUCKET"):
        Remote()


def test_safetensors_invalid_header_type(tmp_path):
    path = tmp_path / "bad.safetensors"
    header = json.dumps([]).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(ValueError):
        _read_safetensors_layout(path)


def test_compose_missing_header_entry():
    header = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    with pytest.raises(KeyError):
        _compose_safetensors(header, ["missing"], {"missing": b"1234"})


def test_materialize_verify_failure(tmp_path):
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"8\x00\x00\x00\x00\x00\x00\x00{}")
    meta = MagicMock()
    meta.storage_unit = "safetensors_v1"
    meta.unit_manifest = {"storage_unit": "safetensors_v1", "header": {}, "tensor_order": [], "parts": []}
    meta.file_checksum_sha256 = "id"
    remote = MagicMock()
    with patch.object(SafetensorsHandler, "restore"):
        with patch.object(SafetensorsHandler, "verify_local", return_value=False):
            with pytest.raises(RuntimeError, match="verification"):
                _materialize_safetensors(remote, meta, tmp_path / "out.safetensors", force=True)


def test_load_remote_index_retries_then_raises():
    remote = MagicMock()
    remote.s3_client.get_object.side_effect = ClientError(
        {"Error": {"Code": "503", "Message": "slow"}}, "GetObject"
    )
    remote.s3_client.exceptions.ClientError = ClientError
    with patch("express.base.remote.time.sleep"):
        with pytest.raises(RuntimeError, match="retries"):
            _load_remote_index(remote, __import__("express.base.data", fromlist=["Torrent"]).Torrent("x"), retries=2)


def test_delete_with_shared_chunks_message():
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.return_value = {}
    meta = __import__("express.base.model", fromlist=["Metadata"]).Metadata(name="m", version="0.1.0")
    torrent = __import__("express.base.data", fromlist=["get_torrent"]).get_torrent(meta)
    owned = {"shared", "exclusive"}
    index = {
        "folder_index": [
            {
                "file_name": "a.bin",
                "file_checksum_sha256": "exclusive",
                "file_relative_path": "a.bin",
            }
        ]
    }
    other_index = {
        "folder_index": [
            {
                "file_name": "b.bin",
                "file_checksum_sha256": "shared",
                "file_relative_path": "b.bin",
            }
        ]
    }
    remote.s3_client.get_object.side_effect = [
        {"Body": MagicMock(read=lambda: json.dumps(index).encode("utf-8"))},
        {"Body": MagicMock(read=lambda: json.dumps(other_index).encode("utf-8"))},
    ]
    page_iterator = MagicMock()
    page_iterator.search.return_value = iter(["other.express-index.json"])
    paginator = MagicMock()
    paginator.paginate.side_effect = [
        page_iterator,
        [{"Contents": [{"Key": "exclusive", "Size": 1}, {"Key": "shared", "Size": 2}]}],
        [{"Contents": [{"Key": "exclusive", "Size": 1}, {"Key": "shared", "Size": 2}]}],
    ]
    remote.s3_client.get_paginator.return_value = paginator
    delete(torrent, remote, yes=True, dry_run=False)


def test_delete_confirms_and_runs():
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.return_value = {}
    meta = __import__("express.base.model", fromlist=["Metadata"]).Metadata(name="m", version="0.1.0")
    torrent = __import__("express.base.data", fromlist=["get_torrent"]).get_torrent(meta)
    index = {"folder_index": []}
    remote.s3_client.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps(index).encode("utf-8"))
    }
    page_iterator = MagicMock()
    page_iterator.search.return_value = iter([])
    paginator = MagicMock()
    paginator.paginate.side_effect = [page_iterator, [{"Contents": [{"Key": "k", "Size": 3}]}]]
    remote.s3_client.get_paginator.return_value = paginator
    delete(torrent, remote, yes=True, dry_run=False)
    remote.s3_client.delete_objects.assert_called()


def test_catalog_populates_entries():
    remote = MagicMock()
    page_iterator = MagicMock()
    page_iterator.search.return_value = iter(["abc.express-index.json"])
    paginator = MagicMock()
    paginator.paginate.return_value = page_iterator
    remote.s3_client.get_paginator.return_value = paginator
    meta = __import__("express.base.model", fromlist=["Metadata"]).Metadata(name="m", version="0.1.0")
    with patch("express.base.remote.get_metadata", return_value=meta):
        entries = catalog(remote)
    assert len(entries) == 1


def test_transfer_session_exception_path():
    with pytest.raises(RuntimeError):
        with TransferSession("X", total_files=1):
            raise RuntimeError("boom")


def test_pull_skip_with_session_on_hash_mismatch(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"local")
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    with TransferSession("Pull", total_files=1) as session:
        pull_file(remote, Path("k"), path, "other", force=False, session=session)
    remote.s3_client.download_file.assert_not_called()


def test_push_chunk_existing_with_session():
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.return_value = {}
    with TransferSession("Push", total_files=1) as session:
        push_chunk(remote, Path(__file__), "exists", session=session)
