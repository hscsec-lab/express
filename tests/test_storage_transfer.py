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
