import json
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from express.base.data import get_torrent
from express.base.model import Metadata
from express.base.remote import _print_entries, delete


def test_delete_refuses_without_yes():
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.return_value = {}
    meta = Metadata(name="m", version="0.1.0")
    torrent = get_torrent(meta)
    index = {"folder_index": []}
    remote.s3_client.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps(index).encode("utf-8"))
    }
    page_iterator = MagicMock()
    page_iterator.search.return_value = iter(["other.express-index.json"])
    paginator = MagicMock()
    paginator.paginate.side_effect = [
        page_iterator,
        [{"Contents": [{"Key": "shared", "Size": 1}]}],
        [{"Contents": [{"Key": "shared", "Size": 1}]}],
    ]
    remote.s3_client.get_paginator.return_value = paginator
    remote.s3_client.get_object.side_effect = [
        {"Body": MagicMock(read=lambda: json.dumps(index).encode("utf-8"))},
        {"Body": MagicMock(read=lambda: json.dumps({"folder_index": []}).encode("utf-8"))},
    ]
    delete(torrent, remote, yes=False, dry_run=False)
    remote.s3_client.delete_objects.assert_not_called()


def test_print_entries_empty(capsys):
    _print_entries([], empty_message="nothing")
    assert "nothing" in capsys.readouterr().out
