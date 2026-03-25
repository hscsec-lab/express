import json
import os
import tempfile
from pathlib import Path
from typing import List

from src import console
from base.client import Remote, search_extension, is_remote_file_exists
from base.data import Torrent, get_torrent, from_torrent, search_exact
from base.file import FolderIndex, FileMetadata
from base.model import Model, Metadata, get_metadata, MODEL_INDEX_FILE_NAME
from base.transfer import push_file, pull_file, open_remote_file

LOCAL_WORKDIR = Path(os.getenv('LOCAL_WORKDIR', '/models'))

def _list(remote: Remote, torrent: bool = False) -> List[Metadata] | List[Torrent]:
    """Retrieves a list of available remote models or their torrents."""
    file_list = search_extension(remote, MODEL_INDEX_FILE_NAME)
    metadata_list: List[Metadata] = []
    torrents: List[Torrent] = []

    for file_name in file_list:
        _torrent = Torrent(file_name.split(f'.{MODEL_INDEX_FILE_NAME}')[0])
        torrents.append(_torrent)
        metadata_list.append(get_metadata(_torrent))

    if torrent:
        return torrents
    return metadata_list # 根据上层调用需求返回Metadata列表或者Torrent列表

def push(model: Model, remote: Remote) -> None:
    """Pushes a model and its indexed files to the remote server."""
    assert model.is_metadata_file_exists(), "The model being pushed is missing metadata."
    model_metadata_torrent: Torrent = get_torrent(model.get_metadata())
    folder_index: List[FileMetadata] = model.folder_index.folder_index

    for file_metadata in folder_index:
        push_file(model, remote, file_metadata, model_metadata_torrent)

    print(f"推送完成，请妥善保存模型torrent: {model_metadata_torrent}")

def pull_model_with_index(index: dict, remote: Remote, save_path: Path, force: bool) -> Model:
    """Pulls all model files defined in the index dictionary to the specified path."""
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
    """Downloads and loads the JSON index content associated with a specific torrent."""
    remote_index_path = f"{torrent}.{MODEL_INDEX_FILE_NAME}"
    assert is_remote_file_exists(remote.s3_client, remote.s3_bucket, remote_index_path), "Remote index is not exist."

    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tmp:
        with open_remote_file(remote, Path(remote_index_path), Path(tmp.name), file_checksum_sha256=None) as pulled_file:
            return json.loads(pulled_file.read())

def pull_model(torrent: Torrent, remote: Remote, model: Model, force: bool) -> Model:
    """Coordinates downloading a model index and sequentially pulling all its files."""
    index = pull_index_with_torrent(torrent, remote)
    return pull_model_with_index(index, remote, save_path=model.path, force=force)

def pull(torrent: Torrent, remote: Remote, force: bool = False) -> Model:
    """Main entrypoint to pull a remote model to the local work directory."""
    metadata: Metadata = from_torrent(torrent, Metadata)
    model_name = metadata.name
    local_model_path = LOCAL_WORKDIR / model_name

    if not local_model_path.exists():
        local_model_path.mkdir(parents=True, exist_ok=True)

    model = Model(path=local_model_path)
    return pull_model(torrent, remote, model, force)

def ls(remote: Remote, torrent: bool = False) -> None:
    """Prints the list of remote models to the console."""
    console.print(_list(remote, torrent), overflow="ignore")

def search(remote: Remote, field: str, value: str) -> None:
    """Searches for and prints models matching a specific metadata field and value."""
    metadata_list = search_exact(_list(remote), field, value)
    console.print(metadata_list)

if __name__ == '__main__':
    ls(remote=Remote())