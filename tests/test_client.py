from unittest.mock import MagicMock, patch

import pytest

from express.base.client import Remote, delete_objects, is_remote_file_exists, list_objects, search_extension


def test_is_remote_file_exists_true():
    client = MagicMock()
    client.head_object.return_value = {}
    assert is_remote_file_exists(client, "b", "k") is True


def test_is_remote_file_exists_missing():
    from botocore.exceptions import ClientError

    client = MagicMock()
    client.exceptions.ClientError = ClientError
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    assert is_remote_file_exists(client, "b", "k") is False


def test_list_objects():
    remote = MagicMock()
    page = {"Contents": [{"Key": "a.express-index.json", "Size": 1}, {"Key": "b.bin", "Size": 2}]}
    remote.s3_client.get_paginator.return_value.paginate.return_value = [page]
    objs = list_objects(remote)
    assert len(objs) == 2


def test_search_extension():
    remote = MagicMock()
    paginator = MagicMock()
    remote.s3_client.get_paginator.return_value = paginator
    paginator.paginate.return_value.search.return_value = iter(["idx.express-index.json", None])
    assert search_extension(remote, "express-index.json") == ["idx.express-index.json"]


def test_is_remote_file_exists_reraises():
    from botocore.exceptions import ClientError

    client = MagicMock()
    client.exceptions.ClientError = ClientError
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "500", "Message": "boom"}}, "HeadObject"
    )
    with pytest.raises(ClientError):
        is_remote_file_exists(client, "b", "k")


def test_delete_objects_empty():
    remote = MagicMock()
    delete_objects(remote, [])
    remote.s3_client.delete_objects.assert_not_called()


def test_delete_objects_batches():
    remote = MagicMock()
    keys = [f"k{i}" for i in range(1001)]
    delete_objects(remote, keys)
    assert remote.s3_client.delete_objects.call_count == 2


@patch.dict("os.environ", {"S3_AK": "a", "S3_SK": "s", "S3_ENDPOINT": "https://x", "S3_BUCKET": "b"}, clear=True)
@patch("express.base.config.ensure_remote_config")
@patch("boto3.client")
@patch("boto3.resource")
def test_remote_init(mock_resource, mock_client, _ensure):
    remote = Remote()
    assert remote.s3_bucket == "b"
    mock_client.assert_called_once()
    mock_resource.assert_called_once()
    assert mock_client.call_args.kwargs.get("config") is not None
    assert mock_resource.call_args.kwargs.get("config") is not None


@patch.dict("os.environ", {}, clear=True)
@patch("express.base.config.ensure_remote_config", side_effect=EnvironmentError("missing"))
def test_remote_missing_env(_ensure):
    with pytest.raises(EnvironmentError):
        Remote()
