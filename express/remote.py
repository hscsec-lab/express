import json
import os
import tempfile
from contextlib import contextmanager
from io import BufferedReader
from pathlib import Path
from typing import List, Any, Generator

import boto3
from rich.text import Text
from simple_file_checksum import get_checksum

from express import console
from express.data import Torrent, get_torrent, from_torrent, search_exact
from express.file import FolderIndex, FileMetadata
from express.model import Model, Metadata, get_metadata, MODEL_INDEX_FILE_NAME
from express.s3 import ProgressPercentage, DownloadProgressSimple

LOCAL_WORKDIR = Path(os.getenv('LOCAL_WORKDIR','/models'))

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

def is_remote_file_exists(s3_client, bucket: str, key: str) -> bool:
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

    if is_remote_file_exists(remote.s3_client, remote.s3_bucket, remote_file_name):
        console.print(Text.assemble("✓ Skip ", (remote_file_name, "dim"), ": already exists"))
        return

    remote.s3_client.upload_file(
        str(local_file_path),
        remote.s3_bucket,
        remote_file_name,
        Callback=ProgressPercentage(str(local_file_path))
    )

def pull_file(
        remote: Remote,
        remote_file_path: Path,
        local_file_path: Path,
        file_checksum_sha256: str | None,
        force=False
) -> Path:
    if local_file_path.exists():
        if file_checksum_sha256:
            local_checksum = get_checksum(local_file_path)
            if local_checksum == file_checksum_sha256:
                console.print(Text.assemble("✓ Skip ", (str(local_file_path), "dim"), ": already exists"))
                return local_file_path
            else:
                if force:
                    console.print(Text.assemble("⚠️  Overwriting ", str(local_file_path), ": checksum mismatch"))
                else:
                    console.print(Text.assemble("✗ Skip ", str(local_file_path),
                                                ": checksum mismatch (use --force to overwrite)"))
                    return local_file_path
    local_file_path.parent.mkdir(parents=True, exist_ok=True)
    remote.s3_client.download_file(
        remote.s3_bucket,
        str(remote_file_path),
        local_file_path,
        Callback=DownloadProgressSimple(str(local_file_path.name))
    )
    return local_file_path

@contextmanager
def open_remote_file(
        remote: Remote,
        remote_file_path: Path,
        local_file_path: Path,
        file_checksum_sha256: str | None,
        force=False
) -> Generator[BufferedReader, Any, None]:
    pull_file(remote,remote_file_path,local_file_path,file_checksum_sha256,force)
    f = None
    try:
        f = local_file_path.open("rb")  # 或 "r"，按需
        yield f
    finally:
        if f:
            f.close()
def search_extension(remote: Remote, extension: str) -> List[str]:
    """

    :param remote:
    :param extension:
    :return:
    """
    paginator = remote.s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=remote.s3_bucket)
    objects = page_iterator.search(f"Contents[?ends_with(Key, `{extension}`)][].Key")

    found_files = []
    for item in objects:
        if item:  # S3中如果空桶会返回[None]而并非[]
            found_files.append(item)

    return found_files


def _list(remote: Remote,torrent: bool = False) -> List[Metadata] | List[Torrent]:
    file_list = search_extension(remote, MODEL_INDEX_FILE_NAME)
    metadatas: List[Metadata] = []
    torrents: List[Torrent] = []
    for file_name in file_list:
        _torrent = Torrent(file_name.split(f'.{MODEL_INDEX_FILE_NAME}')[0])
        torrents.append(_torrent)
        metadatas.append(get_metadata(_torrent))
    if torrent:
        return torrents
    return metadatas


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


def pull_model_with_index(index: dict, remote: Remote, save_path: Path, force: bool) -> Model:
    folder_index: List[FileMetadata] = FolderIndex(**index).folder_index
    for file_metadata in folder_index:
        if MODEL_INDEX_FILE_NAME != file_metadata.file_name:
            download_path = save_path / file_metadata.file_relative_path
            pull_file(
                remote,
                Path(file_metadata.file_checksum_sha256),
                download_path,
                file_checksum_sha256=file_metadata.file_checksum_sha256,
                force=force
            )

    return Model(path=save_path)


def pull_index_with_torrent(torrent: Torrent, remote: Remote) -> dict:
    """
    下载index并返回index的内容
    :param torrent:
    :param remote:
    :return:
    """
    remote_index_path = f"{torrent}.{MODEL_INDEX_FILE_NAME}"
    assert is_remote_file_exists(remote.s3_client, remote.s3_bucket, remote_index_path), "Remote index is not exist."
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tmp:
        with open_remote_file(remote, Path(remote_index_path), Path(tmp.name), file_checksum_sha256=None) as pulled_file:
            return json.loads(pulled_file.read())

def pull_model(torrent: Torrent,remote: Remote,model: Model,force: bool) -> Model:
    index = pull_index_with_torrent(torrent,remote)
    return pull_model_with_index(index,remote,save_path=model.path,force=force)

def pull(torrent: Torrent, remote: Remote, force=False) -> Model:
    """
    从远端拉取模型
    :param torrent:
    :param remote:
    :param force:
    :return:
    """
    metadata: Metadata = from_torrent(torrent, Metadata)
    model_name = metadata.name
    local_model_path = LOCAL_WORKDIR / model_name
    if not local_model_path.exists():
        local_model_path.mkdir(parents=True, exist_ok=True)
    model = Model(path=local_model_path) # 初始化一个空的模型
    return pull_model(torrent,remote,model,force)


def ls(remote: Remote,torrent: bool = False):
    console.print(_list(remote,torrent),overflow="ignore")

def search(remote: Remote, field: str, value: str):
    metadata_list = search_exact(_list(remote), field, value)
    console.print(metadata_list)


if __name__ == '__main__':
    ls(remote=Remote())