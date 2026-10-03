import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from botocore.exceptions import ClientError
from safetensors.numpy import save_file

from express.base.storage import transfer_ops
from express.base.storage.registry import build_file_metadata
from express.base.storage.transfer_ops import _UploadSlotPool, upload_storage_parts
from express.base.progress import TransferSession


def test_upload_storage_parts_runs_blobs_in_parallel(tmp_path):
    path = tmp_path / "t.safetensors"
    tensors = {f"t{i}": np.ones(64, dtype=np.float32) * i for i in range(6)}
    save_file(tensors, path)
    meta = build_file_metadata(path, path.name)

    remote = MagicMock()
    remote.s3_bucket = "bucket"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )

    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def upload_fileobj(*args, **kwargs):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1

    remote.s3_client.upload_fileobj.side_effect = upload_fileobj

    upload_storage_parts(remote, path, meta, blob_concurrency=4)
    assert remote.s3_client.upload_fileobj.call_count == 6
    assert peak >= 2


def test_upload_storage_parts_serial_when_concurrency_one(tmp_path):
    path = tmp_path / "t.safetensors"
    save_file(
        {"a": np.ones(2, dtype=np.float32), "b": np.ones(3, dtype=np.float32)},
        path,
    )
    meta = build_file_metadata(path, path.name)
    remote = MagicMock()
    remote.s3_bucket = "bucket"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )

    upload_storage_parts(remote, path, meta, blob_concurrency=1)
    assert remote.s3_client.upload_fileobj.call_count == 2


def test_upload_slot_pool_blocks_until_slot_freed():
    with TransferSession("Push", total_files=1) as session:
        pool = _UploadSlotPool(session, 1)
        order: list[str] = []

        def worker() -> None:
            with pool.occupy("tensor.b", 10):
                order.append("worker_in")
                time.sleep(0.05)
                order.append("worker_out")

        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.01)
        with pool.occupy("tensor.a", 5):
            order.append("main_in")
        thread.join(timeout=2.0)
        assert order == ["worker_in", "worker_out", "main_in"]


def test_parallel_upload_keyboard_interrupt_shuts_down_pool(tmp_path, monkeypatch):
    from concurrent.futures import Future

    path = tmp_path / "t.safetensors"
    tensors = {f"t{i}": np.full(32, i, dtype=np.float32) for i in range(4)}
    save_file(tensors, path)
    meta = build_file_metadata(path, path.name)

    remote = MagicMock()
    remote.s3_bucket = "bucket"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )

    shutdown_calls: list[tuple[bool, bool]] = []
    fail = Future()
    fail.set_exception(KeyboardInterrupt())

    class FakePool:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers

        def submit(self, fn, *args, **kwargs):
            return fail

        def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(transfer_ops, "ThreadPoolExecutor", FakePool)

    with pytest.raises(KeyboardInterrupt):
        upload_storage_parts(remote, path, meta, blob_concurrency=4)

    assert shutdown_calls == [(False, True)]
