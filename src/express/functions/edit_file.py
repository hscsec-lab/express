import os
import shlex
import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from express import console

from express.base.client import Remote
from express.base.data import Torrent
from express.base.file import FolderIndex, get_remote_chunk_metadata_from_index, fast_checksum
from express.base.model import MODEL_INDEX_FILE_NAME
from express.base.remote import pull_index_with_torrent
from express.base.transfer import push_chunk, pull_file

def open_editor(f):
    """
    Edit a local file using the system's default editor.
    :param f:
    :return:
    """
    editor_cmd = os.getenv('EDITOR', 'vim')
    cmd = shlex.split(editor_cmd)
    cmd.append(f.name)
    subprocess.run(cmd, check=True)

def edit_file(remote: Remote, torrent: Torrent, remote_file_name: str):
    """
    Use the CLI editing tool to edit files online. Only files that can be decoded as str are allowed to be edited.
    After editing, the file will be automatically uploaded and the hash value of the index file will be updated.
    :param remote:
    :param torrent:
    :param remote_file_name: Remote file name, must be a file that exists in the index file
    :return:
    """
    with open_remote_file_rw(remote, torrent, remote_file_name) as f:
        open_editor(f)

@contextmanager
def open_remote_file_rw(remote: Remote, torrent: Torrent, remote_file_name: str):
    """
    Modify the target file and update the index file
    :param remote:
    :param torrent:
    :param remote_file_name:
    :return:
    """
    index: FolderIndex = FolderIndex(**pull_index_with_torrent(torrent=torrent, remote=remote))
    remote_chunk_metadata = get_remote_chunk_metadata_from_index(index, remote_file_name)
    assert remote_file_name in [index_file.file_name for index_file in index.folder_index], f"File {remote_file_name} is not exist in index."
    with NamedTemporaryFile('r+') as metadata_file:
        with NamedTemporaryFile('r+') as tmp_file:
            pull_file(remote,Path(remote_chunk_metadata.file_checksum_sha256), Path(tmp_file.name), file_checksum_sha256=remote_chunk_metadata.file_checksum_sha256, force=True)
            try:
                with open(tmp_file.name, 'r+', encoding='utf-8') as f:
                    try:
                        yield f
                    finally:
                        f.flush()
                        os.fsync(f.fileno())
            finally:
                remote_chunk_hash = fast_checksum(Path(tmp_file.name))
                if remote_chunk_metadata.file_checksum_sha256 != remote_chunk_hash:
                    console.print(f"File {remote_file_name} has been modified, syncing changes...")
                    remote_chunk_metadata.file_checksum_sha256 = remote_chunk_hash
                    push_chunk(remote=remote,local_file_path=Path(tmp_file.name),force=True)
                    index_json_str = index.model_dump_json()
                    metadata_file.write(index_json_str)
                    metadata_file.flush()
                    push_chunk(remote=remote,local_file_path=Path(metadata_file.name),remote_file_name=f"{torrent}.{MODEL_INDEX_FILE_NAME}",force=True)
                    console.print(f"File has been synced successfully.")
                else:
                    console.print(f"No changes detected in {remote_file_name}, skipping sync.")
