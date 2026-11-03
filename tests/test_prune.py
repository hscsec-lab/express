import json
from unittest.mock import MagicMock

import pytest

from express.base.model import MODEL_INDEX_FILE_NAME
from express.base.remote import prune


def _index(*files):
    return {"folder_index": list(files)}


def _plain(name, checksum):
    return {
        "file_name": name,
        "file_checksum_sha256": checksum,
        "file_relative_path": name,
    }


def _bucket(index_payloads, objects):
    remote = MagicMock()
    remote.s3_bucket = "b"

    def get_object(Bucket, Key):
        payload = index_payloads[Key]
        return {"Body": MagicMock(read=lambda payload=payload: json.dumps(payload).encode("utf-8"))}

    remote.s3_client.get_object.side_effect = get_object
    page_iterator = MagicMock()
    page_iterator.search.return_value = iter(index_payloads)
    paginator = MagicMock()
    paginator.paginate.side_effect = [
        page_iterator,
        [{"Contents": objects}],
    ]
    remote.s3_client.get_paginator.return_value = paginator
    return remote


def _deleted_keys(remote):
    remote.s3_client.delete_objects.assert_called_once()
    batch = remote.s3_client.delete_objects.call_args.kwargs["Delete"]["Objects"]
    return {item["Key"] for item in batch}


def test_prune_requires_confirmation():
    index_key = f"torrent.{MODEL_INDEX_FILE_NAME}"
    remote = _bucket(
        {index_key: _index(_plain("w.bin", "keep"))},
        [
            {"Key": index_key, "Size": 10},
            {"Key": "keep", "Size": 100},
            {"Key": "orphan-a", "Size": 5},
            {"Key": "orphan-b", "Size": 7},
        ],
    )
    seen = []

    def confirm(message):
        seen.append(message)
        return False

    prune(remote, confirm=confirm)
    assert seen == ["Delete 2 unreferenced chunks (12.0 B, 12 bytes)?"]
    remote.s3_client.delete_objects.assert_not_called()


def test_prune_deletes_only_unreferenced_chunks_after_confirm():
    index_key = f"torrent.{MODEL_INDEX_FILE_NAME}"
    remote = _bucket(
        {
            index_key: _index({
                "file_name": "m.safetensors",
                "file_checksum_sha256": "whole-file",
                "file_relative_path": "m.safetensors",
                "storage_unit": "safetensors_v1",
                "unit_manifest": {"parts": [{"content_hash": "blob-1"}, {"content_hash": "blob-2"}]},
            })
        },
        [
            {"Key": index_key, "Size": 20},
            {"Key": "blob-1", "Size": 1000},
            {"Key": "blob-2", "Size": 2000},
            {"Key": "whole-file", "Size": 4},
            {"Key": "stale", "Size": 8},
        ],
    )
    prune(remote, confirm=lambda _message: True)
    assert _deleted_keys(remote) == {"whole-file", "stale"}


def test_prune_yes_skips_prompt(capsys):
    remote = _bucket({}, [{"Key": "stale", "Size": 3}])
    prune(remote, yes=True, confirm=lambda _message: False)
    assert _deleted_keys(remote) == {"stale"}
    assert "3 bytes" in capsys.readouterr().out


def test_prune_dry_run_does_not_delete(capsys):
    remote = _bucket({}, [{"Key": "stale", "Size": 3}])
    prune(remote, dry_run=True, confirm=lambda _message: True)
    remote.s3_client.delete_objects.assert_not_called()
    assert "Dry run: no objects deleted." in capsys.readouterr().out


def test_prune_empty_does_not_prompt():
    remote = _bucket({}, [])
    prune(remote, confirm=lambda _message: (_ for _ in ()).throw(AssertionError("prompted")))
    remote.s3_client.delete_objects.assert_not_called()


def test_prune_aborts_when_index_cannot_be_loaded():
    index_key = f"bad.{MODEL_INDEX_FILE_NAME}"
    remote = MagicMock()
    page_iterator = MagicMock()
    page_iterator.search.return_value = iter([index_key])
    paginator = MagicMock()
    paginator.paginate.return_value = page_iterator
    remote.s3_client.get_paginator.return_value = paginator
    remote.s3_client.get_object.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="Aborting prune"):
        prune(remote, yes=True)
    remote.s3_client.delete_objects.assert_not_called()


def test_prune_aborts_when_listing_has_unscanned_index():
    remote = MagicMock()
    page_iterator = MagicMock()
    page_iterator.search.return_value = iter([])
    paginator = MagicMock()
    paginator.paginate.side_effect = [
        page_iterator,
        [{"Contents": [{"Key": f"x.{MODEL_INDEX_FILE_NAME}", "Size": 1}]}],
    ]
    remote.s3_client.get_paginator.return_value = paginator
    with pytest.raises(RuntimeError, match="not scanned"):
        prune(remote, yes=True)
    remote.s3_client.delete_objects.assert_not_called()
