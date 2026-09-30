from pathlib import Path
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from express.base.file import FileMetadata
from express.base.model import MODEL_INDEX_FILE_NAME, Model
from express.base.transfer import pull_file, push_file


def test_push_index_file_branch(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "a.txt").write_text("x", encoding="utf-8")
    model = Model(model_dir)
    model.create_metadata(name="m", version="0.1.0")
    model.write_index_file()
    index_meta = FileMetadata(
        file_name=MODEL_INDEX_FILE_NAME,
        file_checksum_sha256="ignored",
        file_relative_path=Path(MODEL_INDEX_FILE_NAME),
    )
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    remote.s3_client.exceptions.ClientError = ClientError
    remote.s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    push_file(model, remote, index_meta, "torrent123")
    assert remote.s3_client.upload_file.call_args[0][2].endswith(MODEL_INDEX_FILE_NAME)


def test_pull_match_skips_with_session(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"same")
    from express.base.file import fast_checksum

    digest = fast_checksum(path)
    remote = MagicMock()
    remote.s3_bucket = "b"
    remote.s3_client = MagicMock()
    from express.base.progress import TransferSession

    with TransferSession("Pull", total_files=1) as session:
        pull_file(remote, Path(digest), path, digest, force=False, session=session)
    remote.s3_client.download_file.assert_not_called()
