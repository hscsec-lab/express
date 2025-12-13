import json
import os
from pathlib import Path
from typing import List, Optional

import boto3
from simple_file_checksum import get_checksum

from express.data import Torrent, get_torrent, from_torrent
from express.file import FolderIndex, FileMetadata
from express.model import Model, Metadata
from express.s3 import ProgressPercentage, DownloadProgressSimple


class Remote:
    def __init__(self):
        for var in ["S3_AK", "S3_SK", "S3_ENDPOINT", "S3_BUCKET"]:
            if not os.getenv(var):
                raise EnvironmentError(f"Missing required environment variable: {var}")
        self.s3_ak = os.getenv("S3_AK")
        self.s3_sk = os.getenv("S3_SK")
        self.s3_endpoint = os.getenv("S3_ENDPOINT")
        self.s3_bucket = os.getenv("S3_BUCKET")
        self.s3 = boto3.resource(
            's3',
            aws_access_key_id=self.s3_ak,
            aws_secret_access_key=self.s3_sk,
            endpoint_url=self.s3_endpoint
        )
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.s3_ak,
            aws_secret_access_key=self.s3_sk,
            endpoint_url=self.s3_endpoint
        )


def _remote_file_exists(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise


def push_file(
        model: Model,
        remote: Remote,
        file_metadata: FileMetadata,
        model_metadata_torrent: str
) -> None:
    remote_file_name = file_metadata.file_checksum_sha256
    if file_metadata.file_name == model.index_file_name:
        remote_file_name = f"{model_metadata_torrent}.{model.index_file_name}"
    local_file_path = model.path / file_metadata.file_relative_path

    if _remote_file_exists(remote.s3_client, remote.s3_bucket, remote_file_name):
        print(f"✓ Skip {remote_file_name}: already exists")
        return

    remote.s3_client.upload_file(
        str(local_file_path),
        remote.s3_bucket,
        remote_file_name,
        Callback=ProgressPercentage(str(local_file_path))
    )


def push(model: Model, remote: Remote):
    """
    推送模型
    :return:
    """
    assert model.is_metadata_file_exists(), "The model being pushed is missing metadata."
    model_metadata_torrent: Torrent = get_torrent(model.get_metadata())

    folder_index: List[FileMetadata] = model.folder_index.folder_index

    for file_metadata in folder_index:
        push_file(model, remote, file_metadata, model_metadata_torrent)
    print(f"推送完成，请妥善保存模型torrent: {model_metadata_torrent}")


def pull_file(
        remote: Remote,
        remote_file_path: Path,
        local_file_path: Path,
        file_checksum_sha256: str | None,
        force = False
) -> None:
    if local_file_path.exists():
        if file_checksum_sha256:
            if get_checksum(local_file_path) == file_checksum_sha256:
                print(f"✓ Skip {local_file_path}: already exists")
            else:
                if force:
                    print(f"⚠️  Overwriting {local_file_path}: checksum mismatch")
                else:
                    print(f"✗ Skip {local_file_path}: checksum mismatch (use --force to overwrite)")

    local_file_path.parent.mkdir(parents=True, exist_ok=True)
    remote.s3_client.download_file(
        remote.s3_bucket,
        str(remote_file_path),
        local_file_path,
        Callback=DownloadProgressSimple(str(local_file_path))
    )
    print(f"✓ Successfully downloaded {local_file_path}.")


def pull(torrent: Torrent, remote: Remote,force = False) -> Model:
    """
    从远端拉取模型
    :param torrent:
    :param remote:
    :param force:
    :return:
    """
    metadata: Metadata = from_torrent(torrent, Metadata)
    local_model_path = Path(metadata.name)
    if not local_model_path.exists():
        local_model_path.mkdir()
    model = Model(path=local_model_path)
    local_index_file_path = model.path / model.index_file_name
    remote_index_path = f"{torrent}.{model.index_file_name}"
    assert _remote_file_exists(remote.s3_client, remote.s3_bucket, remote_index_path), "Remote index is not exist."
    pull_file(remote, Path(remote_index_path), local_index_file_path,file_checksum_sha256=None)  # 从远端覆写index
    with local_index_file_path.open('r', encoding='utf-8') as f:
        folder_index: List[FileMetadata] = FolderIndex(**json.loads(f.read())).folder_index
        for file_metadata in folder_index:
            pull_file(
                remote,
                file_metadata.file_checksum_sha256,
                model.path / file_metadata.file_relative_path,
                file_checksum_sha256=file_metadata.file_checksum_sha256,
                force=force
            )


if __name__ == '__main__':
    # push(model=Model(Path('../test_folder')),remote=Remote())
    pull(
        "789cab564a2c2dc9c82f2a56b2524aad48cc2dc8498d8789e828a5e62666e680a4324a7313f31cc0a45e727e2e50aa2cb5a838333f0f2867a00784409192c474a0d268b83140be52ac8e525e626e2a50554a6a5a62694e89522d00e029268a",
        Remote())
