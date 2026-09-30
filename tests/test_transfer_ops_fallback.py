from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from botocore.exceptions import ClientError
from safetensors.numpy import save_file

from express.base.file import FileMetadata
from express.base.storage.transfer_ops import upload_storage_parts


def test_upload_reprocesses_when_manifest_mismatch(tmp_path):
    path = tmp_path / "m.safetensors"
    save_file({"w": np.array([1.0], dtype=np.float32)}, path)
    stale = FileMetadata(
        file_name="m.safetensors",
        file_checksum_sha256="dead",
        file_relative_path=Path("m.safetensors"),
        storage_unit="safetensors_v1",
        unit_manifest={
            "storage_unit": "safetensors_v1",
            "header": {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
            "tensor_order": ["w"],
            "parts": [{"name": "w", "content_hash": "bad", "byte_length": 4}],
        },
    )
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    upload_storage_parts(remote, path, stale)
    assert remote.s3_client.upload_fileobj.called
