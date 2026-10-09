from pathlib import Path
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from express.base.storage.base import StorageBlob
from express.base.transfer import push_storage_blob


def test_push_storage_blob_skip_with_session():
    from express.base.progress import TransferSession

    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.return_value = {"ContentLength": 1}
    blob = StorageBlob(content_hash="abc", data=b"x")
    with TransferSession("Push", total_files=1) as session:
        push_storage_blob(remote, blob, "abc", session=session)
    remote.s3_client.upload_fileobj.assert_not_called()


def test_push_storage_blob_skip_existing():
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.return_value = {"ContentLength": 1}
    blob = StorageBlob(content_hash="abc", data=b"x")
    push_storage_blob(remote, blob, "abc", force=False)
    remote.s3_client.upload_fileobj.assert_not_called()


def test_push_storage_blob_decomposed_skip_slot():
    from express.base.progress import TransferSession

    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.return_value = {"ContentLength": 1}
    blob = StorageBlob(content_hash="abc", data=b"x", byte_length=1)
    with TransferSession("Push", total_files=1) as session:
        session.begin_decomposed_file("m.safetensors", 1, slot_count=1)
        session.on_slot_start(0, "t.weight", 1)
        push_storage_blob(
            remote,
            blob,
            "abc",
            session=session,
            decomposed_session=session,
            slot=0,
        )
    remote.s3_client.upload_fileobj.assert_not_called()


def test_push_storage_blob_decomposed_inline_bytes():
    from express.base.progress import TransferSession

    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    blob = StorageBlob(content_hash="k", data=b"inline-bytes")
    with TransferSession("Push", total_files=1) as session:
        session.begin_decomposed_file("m.safetensors", len(b"inline-bytes"), slot_count=1)
        session.on_slot_start(0, "t.bias", len(b"inline-bytes"))
        push_storage_blob(
            remote,
            blob,
            "k",
            session=session,
            decomposed_session=session,
            slot=0,
        )
    remote.s3_client.upload_file.assert_called_once()


def test_push_storage_blob_bytes_fallback():
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    blob = StorageBlob(content_hash="k", data=b"inline")
    push_storage_blob(remote, blob, "k")
    remote.s3_client.upload_file.assert_called_once()
