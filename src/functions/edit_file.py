import os
import shlex
import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from simple_file_checksum import get_checksum
from src import console

from base.client import Remote
from base.data import Torrent
from base.file import FolderIndex, get_remote_chunk_metadata_from_index
from base.model import MODEL_INDEX_FILE_NAME
from base.remote import pull_index_with_torrent
from base.transfer import push_chunk, pull_file


def edit_file(remote: Remote, torrent: Torrent, remote_file_name: str):
    """
    使用cli的编辑工具在线编辑文件，只允许编辑可以被decode为str的文件，编辑完成后会自动上传并更新索引文件的hash值。
    :param remote:
    :param torrent:
    :param remote_file_name: 远程文件名，必须是索引文件中存在的文件
    :return:
    """
    with open_remote_file_rw(remote, torrent, remote_file_name) as f:
        editor_cmd = os.getenv('EDITOR', 'vim')
        cmd = shlex.split(editor_cmd)
        cmd.append(f.name)
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print(f"错误：找不到编辑器 '{cmd[0]}'")

@contextmanager
def open_remote_file_rw(remote: Remote, torrent: Torrent, remote_file_name: str):
    """
    在目标文件上进行修改，并更新索引文件
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
                remote_chunk_hash = get_checksum(tmp_file.name)
                if remote_chunk_metadata.file_checksum_sha256 != remote_chunk_hash:
                    console.print(f"File {remote_file_name} has been modified, syncing changes...")
                    remote_chunk_metadata.file_checksum_sha256 = remote_chunk_hash # 更新索引文件中的hash值
                    push_chunk(remote=remote,local_file_path=Path(tmp_file.name),force=True) # 更新远程修改文件
                    index_json_str = index.model_dump_json()
                    metadata_file.write(index_json_str)
                    metadata_file.flush()
                    push_chunk(remote=remote,local_file_path=Path(metadata_file.name),remote_file_name=f"{torrent}.{MODEL_INDEX_FILE_NAME}",force=True) # 更新远程索引文件
                    console.print(f"File has been synced successfully.")
                else:
                    console.print(f"No changes detected in {remote_file_name}, skipping sync.")
if __name__ == '__main__':
    torrent = Torrent(
        "789cab564a2c2dc9c82f2a56b2524aad48cc2dc8498d8789e828a5e62666e680a4324a7313f31cc0a45e727e2e50aa2cb5a838333f0f2867a00784409192c474a0d268b83140be52ac8e525e626e2a505560796a9eb1ae819e9993be522d0075cd26a5")
    edit_file(Remote(), torrent, "config.json")
