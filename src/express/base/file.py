import hashlib
from pathlib import Path
from typing import List

from pydantic import BaseModel, field_serializer, field_validator


class FileMetadata(BaseModel):
    file_name: str
    file_checksum_sha256: str
    file_relative_path: Path

    @field_serializer('file_relative_path')
    def serialize_path(self, v: Path) -> str:
        return v.as_posix()

    @field_validator('file_relative_path', mode='before')
    @classmethod
    def parse_path(cls, v) -> Path:
        if isinstance(v, Path):
            return v
        elif isinstance(v, str):
            return Path(v)
        else:
            raise ValueError(f"Expected str or Path, got {type(v)}")

    def get_remote_chunk_name(self):
        return self.file_checksum_sha256


class FolderIndex(BaseModel):
    folder_index: List[FileMetadata] = []


def fast_checksum(file_path: Path, sample_size: int = 65536, sample_segments: int = 3) -> str:
    """
    Generate a SHA256 fingerprint by sampling specific segments of a large file.
    """
    stat = file_path.stat()
    file_size = stat.st_size
    hasher = hashlib.sha256()

    if file_size <= sample_size * sample_segments:
        with open(file_path, "rb") as f:
            hasher.update(f.read())
    else:
        with open(file_path, "rb") as f:
            offsets = [
                max(0, min((file_size - sample_size) * i // (sample_segments - 1), file_size - sample_size))
                for i in range(sample_segments)
            ]
            for offset in offsets:
                f.seek(offset)
                hasher.update(f.read(sample_size))

            hasher.update(f"{file_size}_{stat.st_mtime}".encode())

    return hasher.hexdigest()


def generate_index(folder_path: Path) -> FolderIndex:
    """
    根据文件夹生成索引
    :param folder_path:
    :return:
    """
    folder_index: FolderIndex = FolderIndex(folder_index=[])
    file_metadatas: List[FileMetadata] = []
    for file_path in folder_path.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(folder_path)
            file_metadatas.append(
                FileMetadata(
                    file_name=file_path.name,
                    file_checksum_sha256=fast_checksum(file_path),
                    file_relative_path=relative_path
                )
            )
    return FolderIndex(folder_index=file_metadatas)


def get_remote_chunk_metadata_from_index(folder_index: FolderIndex, remote_file_name: str) -> FileMetadata | None:
    """
        根据索引获取远程文件名
    :param folder_index:
    :param remote_file_name:
    :return:
    """
    for file_metadata in folder_index.folder_index:
        if file_metadata.file_name == remote_file_name:
            return file_metadata
    return None
