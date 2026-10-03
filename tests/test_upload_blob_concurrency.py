import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from botocore.exceptions import ClientError
from safetensors.numpy import save_file

from express.base.storage.registry import build_file_metadata
from express.base.storage.transfer_ops import upload_storage_parts


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
