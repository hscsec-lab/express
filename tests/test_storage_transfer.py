from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from safetensors.numpy import save_file

from express.base.file import FileMetadata
from express.base.storage.registry import build_file_metadata
from express.base.storage.transfer_ops import materialize_local_file, upload_storage_parts


def test_upload_storage_parts_deduplicates_identical_tensors(tmp_path):
    path = tmp_path / "t.safetensors"
    save_file({"a": np.ones(2, dtype=np.float32), "b": np.ones(2, dtype=np.float32)}, path)
    meta = build_file_metadata(path, path.name)
    remote = MagicMock()
    remote.s3_bucket = "bucket"
    remote.s3_client = MagicMock()
    from botocore.exceptions import ClientError

    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    upload_storage_parts(remote, path, meta)
    upload_calls = remote.s3_client.upload_fileobj.call_count + remote.s3_client.upload_file.call_count
    assert upload_calls == 1


def test_materialize_safetensors_from_remote(tmp_path):
    path = tmp_path / "m.safetensors"
    save_file({"w": np.array([1.0, 2.0], dtype=np.float32)}, path)
    meta = build_file_metadata(path, path.name)
    processed = __import__(
        "express.base.storage.safetensors", fromlist=["SafetensorsHandler"]
    ).SafetensorsHandler().process(path)
    blobs = {b.content_hash: b.read_payload() for b in processed.blobs}

    remote = MagicMock()
    remote.s3_bucket = "bucket"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = Exception

    def fake_download(bucket, key, filename, Callback=None):
        Path(filename).write_bytes(blobs[key])

    remote.s3_client.download_file.side_effect = fake_download
    remote.s3_client.head_object.side_effect = lambda **kwargs: {
        "ContentLength": len(blobs[kwargs["Key"]])
    }

    from express.base.storage.safetensors import SafetensorsHandler

    dest = tmp_path / "out" / "m.safetensors"
    materialize_local_file(remote, meta, dest, force=True)
    handler = SafetensorsHandler()
    assert handler.verify_local(dest, meta.file_checksum_sha256, meta.unit_manifest)


def test_unique_manifest_parts_drops_duplicate_hash():
    from express.base.storage.transfer_ops import _unique_manifest_parts

    parts = [
        {"content_hash": "a", "name": "a"},
        {"content_hash": "a", "name": "b"},
        {"content_hash": "c", "name": "c"},
    ]
    assert [part["name"] for part in _unique_manifest_parts(parts)] == ["a", "c"]


def test_materialize_safetensors_progress_is_one_file(tmp_path):
    path = tmp_path / "m.safetensors"
    save_file(
        {
            "a": np.ones(4, dtype=np.float32),
            "b": np.zeros(8, dtype=np.float32),
        },
        path,
    )
    meta = build_file_metadata(path, path.name)
    processed = __import__(
        "express.base.storage.safetensors", fromlist=["SafetensorsHandler"]
    ).SafetensorsHandler().process(path)
    blobs = {b.content_hash: b.read_payload() for b in processed.blobs}
    assert len(blobs) == 2

    remote = MagicMock()
    remote.s3_bucket = "bucket"
    remote.s3_client = MagicMock()

    def fake_download(bucket, key, filename, Callback=None):
        payload = blobs[key]
        Path(filename).write_bytes(payload)
        if Callback is not None:
            Callback(len(payload))

    remote.s3_client.download_file.side_effect = fake_download
    from express.base.progress import TransferSession

    dest = tmp_path / "out.safetensors"
    with TransferSession("Pull", total_files=23) as session:
        materialize_local_file(remote, meta, dest, force=True, session=session)
        assert session._files_done == 1
    assert remote.s3_client.head_object.call_count == 0
    assert dest.is_file()


def test_parallel_download_keyboard_interrupt_shuts_down_pool(tmp_path, monkeypatch):
    from concurrent.futures import Future

    import express.base.storage.transfer_ops as transfer_ops

    path = tmp_path / "m.safetensors"
    save_file(
        {f"t{i}": np.full(8, i, dtype=np.float32) for i in range(3)},
        path,
    )
    meta = build_file_metadata(path, path.name)
    remote = MagicMock()
    remote.s3_bucket = "bucket"
    remote.s3_client = MagicMock()
    fail = Future()
    fail.set_exception(KeyboardInterrupt())
    shutdown_calls: list[tuple[bool, bool]] = []

    class FakePool:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers

        def submit(self, fn, *args, **kwargs):
            return fail

        def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(transfer_ops, "ThreadPoolExecutor", FakePool)
    with __import__("pytest").raises(KeyboardInterrupt):
        materialize_local_file(remote, meta, tmp_path / "out.safetensors", force=True, session=None)
    assert shutdown_calls == [(False, True)]
