import json
import os
import tempfile
import time
from pathlib import Path
from typing import List, Set, Tuple

from botocore.exceptions import ClientError
from rich.text import Text

from express import console
from express.base.client import Remote, search_extension, is_remote_file_exists, list_objects, delete_objects
from express.base.data import Torrent, get_torrent, from_torrent, search_exact
from express.base.file import FolderIndex, FileMetadata
from express.base.model import Model, Metadata, get_metadata, MODEL_INDEX_FILE_NAME
from express.base.transfer import push_file, pull_file, open_remote_file

def local_workdir() -> Path:
    from express.base.config import default_local_workdir

    return Path(os.getenv("LOCAL_WORKDIR") or default_local_workdir())


def _format_bytes(num: float) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def _index_remote_key(torrent: Torrent) -> str:
    return f"{torrent}.{MODEL_INDEX_FILE_NAME}"


def _torrent_from_index_key(index_key: str) -> Torrent:
    suffix = f".{MODEL_INDEX_FILE_NAME}"
    if not index_key.endswith(suffix):
        raise ValueError(f"Not an index key: {index_key}")
    return Torrent(index_key[: -len(suffix)])


def _content_keys_from_index(index: FolderIndex) -> Set[str]:
    """Content-addressed object keys for a model (excludes the torrent index object)."""
    keys: Set[str] = set()
    for file_metadata in index.folder_index:
        if file_metadata.file_name == MODEL_INDEX_FILE_NAME:
            continue
        keys.add(file_metadata.file_checksum_sha256)
    return keys


def _load_remote_index(remote: Remote, torrent: Torrent, retries: int = 5) -> FolderIndex:
    """Load a remote folder index via GetObject, with retries for transient S3/R2 errors."""
    index_key = _index_remote_key(torrent)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = remote.s3_client.get_object(Bucket=remote.s3_bucket, Key=index_key)
            payload = json.loads(response["Body"].read())
            return FolderIndex(**payload)
        except ClientError as exc:
            last_error = exc
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise
            time.sleep(min(2 ** attempt, 8))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid remote index JSON: {index_key}") from exc
    raise RuntimeError(f"Failed to load remote index after {retries} retries: {index_key}") from last_error


def _referenced_chunks(remote: Remote, exclude_torrent: Torrent | None = None) -> Set[str]:
    """
    Collect content chunk keys still referenced by other remote model indexes.

    Aborts if any other index cannot be loaded, so shared chunks are never deleted
    based on an incomplete reference scan.
    """
    referenced: Set[str] = set()
    index_keys = search_extension(remote, MODEL_INDEX_FILE_NAME)
    for index_key in index_keys:
        torrent = _torrent_from_index_key(index_key)
        if exclude_torrent is not None and torrent == exclude_torrent:
            continue
        try:
            index = _load_remote_index(remote, torrent)
        except Exception as exc:
            raise RuntimeError(
                f"Aborting delete: cannot verify references from index {index_key}: {exc}"
            ) from exc
        referenced |= _content_keys_from_index(index)
    return referenced


def bucket_stats(remote: Remote) -> Tuple[int, int]:
    """Return (object_count, total_bytes) for the configured bucket."""
    objects = list_objects(remote)
    return len(objects), sum(obj["Size"] for obj in objects)


def du(remote: Remote) -> None:
    """Print total object count and storage size of the remote bucket."""
    count, size = bucket_stats(remote)
    console.print(f"Objects: {count}")
    console.print(f"Size:    {_format_bytes(size)} ({size} bytes)")


def delete(torrent: Torrent, remote: Remote, yes: bool = False, dry_run: bool = False) -> None:
    """
    Delete a remote model identified by torrent.

    Always removes the torrent index. Content chunks are removed only when no other
    remote model index still references them, so shared objects are never orphaned.
    """
    index_key = _index_remote_key(torrent)
    assert is_remote_file_exists(remote.s3_client, remote.s3_bucket, index_key), "Remote index does not exist."

    # Validate torrent payload early so we fail before mutating storage.
    metadata = from_torrent(torrent, Metadata)
    index = _load_remote_index(remote, torrent)
    owned_keys = _content_keys_from_index(index)
    shared_keys = owned_keys & _referenced_chunks(remote, exclude_torrent=torrent)
    exclusive_keys = owned_keys - shared_keys

    size_by_key = {obj["Key"]: obj["Size"] for obj in list_objects(remote)}
    missing_owned = sorted(key for key in owned_keys if key not in size_by_key)
    delete_keys = sorted(exclusive_keys | {index_key})
    delete_bytes = sum(size_by_key.get(key, 0) for key in delete_keys)
    keep_bytes = sum(size_by_key.get(key, 0) for key in shared_keys)

    console.print(Text.assemble("Delete torrent ", (str(torrent)[:48] + "...", "dim")))
    console.print(f"Model:           {metadata.name} ({metadata.version})")
    console.print(f"Index:           {index_key}")
    console.print(f"Owned chunks:    {len(owned_keys)}")
    console.print(f"Exclusive:       {len(exclusive_keys)} ({_format_bytes(sum(size_by_key.get(k, 0) for k in exclusive_keys))})")
    console.print(f"Shared (keep):   {len(shared_keys)} ({_format_bytes(keep_bytes)})")
    if missing_owned:
        console.print(f"Missing remote:  {len(missing_owned)} owned keys already absent")
    console.print(f"Will delete:     {len(delete_keys)} objects ({_format_bytes(delete_bytes)})")

    if dry_run:
        console.print("Dry run: no objects deleted.")
        return

    if not yes:
        console.print("Refusing to delete without --yes / -y confirmation.")
        return

    delete_objects(remote, delete_keys)
    console.print(
        Text.assemble(
            "✓ Deleted ",
            (str(len(delete_keys)), "bold"),
            " objects, freed ",
            (_format_bytes(delete_bytes), "bold green"),
            f" ({delete_bytes} bytes).",
        )
    )
    if shared_keys:
        console.print(
            Text.assemble(
                "✓ Kept ",
                (str(len(shared_keys)), "bold"),
                " shared chunks still referenced by other models.",
            )
        )

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
    return metadata_list # Return Metadata list or Torrent list based on upper-layer call requirements

def push(model: Model, remote: Remote) -> None:
    """Pushes a model and its indexed files to the remote server."""
    assert model.is_metadata_file_exists(), "The model being pushed is missing metadata."
    model_metadata_torrent: Torrent = get_torrent(model.get_metadata())
    folder_index: List[FileMetadata] = model.folder_index.folder_index

    for file_metadata in folder_index:
        push_file(model, remote, file_metadata, model_metadata_torrent)

    print(f"Push completed. Model torrent: {model_metadata_torrent}")

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
    local_model_path = local_workdir() / model_name

    if not local_model_path.exists():
        local_model_path.mkdir(parents=True, exist_ok=True)

    model = Model(path=local_model_path)
    return pull_model(torrent, remote, model, force=force)

def ls(remote: Remote, torrent: bool = False) -> None:
    """Prints the list of remote models to the console."""
    console.print(_list(remote, torrent), overflow="ignore")

def search(remote: Remote, field: str, value: str) -> None:
    """Searches for and prints models matching a specific metadata field and value."""
    metadata_list = search_exact(_list(remote), field, value)
    console.print(metadata_list)

if __name__ == '__main__':
    ls(remote=Remote())