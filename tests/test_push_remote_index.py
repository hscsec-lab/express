from pathlib import Path
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from express.base.data import get_torrent
from express.base.model import MODEL_INDEX_FILE_NAME, Model
from express.base.remote import push


def test_push_uploads_metadata_torrent_index_key(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "a.txt").write_text("payload", encoding="utf-8")
    model = Model(model_dir)
    model.create_metadata(name="m", version="0.1.0", authors="a")

    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )

    push(model, remote)

    torrent = get_torrent(model.get_metadata())
    expected_key = f"{torrent}.{MODEL_INDEX_FILE_NAME}"
    uploaded_keys = [call[0][2] for call in remote.s3_client.upload_file.call_args_list]
    assert expected_key in uploaded_keys
