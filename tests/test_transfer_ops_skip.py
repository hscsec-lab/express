from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from safetensors.numpy import save_file

from express.base.progress import TransferSession
from express.base.storage.registry import build_file_metadata
from express.base.storage.transfer_ops import materialize_local_file


def test_materialize_skip_when_local_matches(tmp_path):
    path = tmp_path / "m.safetensors"
    save_file({"w": np.array([1.0], dtype=np.float32)}, path)
    meta = build_file_metadata(path, path.name)
    remote = MagicMock()
    with TransferSession("Pull", total_files=1) as session:
        out = materialize_local_file(remote, meta, path, force=False, session=session)
    assert out == path
    remote.s3_client.download_file.assert_not_called()
