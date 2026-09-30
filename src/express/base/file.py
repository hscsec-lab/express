import hashlib
from pathlib import Path
from typing import List

from pydantic import BaseModel, field_serializer, field_validator


class FileMetadata(BaseModel):
    file_name: str
    file_checksum_sha256: str
    file_relative_path: Path
    storage_unit: str | None = None
    unit_manifest: dict | None = None

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
    Generate a file index from the given folder.
    :param folder_path:
    :return:
    """
    from express.base.model import MODEL_INDEX_FILE_NAME
    import time

    from express.base.progress import index_progress, short_label
    from express.base.remote import _format_bytes
    from express.base.storage.registry import build_file_metadata

    candidates = sorted(
        (
            path
            for path in folder_path.rglob("*")
            if path.is_file() and path.name != MODEL_INDEX_FILE_NAME
        ),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )

    file_metadatas: List[FileMetadata] = []
    with index_progress() as progress:
        task_id = progress.add_task(
            "Indexing files",
            total=max(len(candidates), 1),
            tensor="",
        )
        for file_path in candidates:
            relative_path = file_path.relative_to(folder_path)
            size_label = _format_bytes(file_path.stat().st_size)
            progress.update(
                task_id,
                description=f"Hash {short_label(file_path.name, 28)} ({size_label})",
                tensor="",
            )
            last_tensor_update = [0.0]

            def on_tensor(name: str, index: int, total: int, *, fname=file_path.name) -> None:
                now = time.monotonic()
                if index != total and now - last_tensor_update[0] < 0.2:
                    return
                last_tensor_update[0] = now
                progress.update(
                    task_id,
                    description=f"Hash {short_label(fname, 28)}",
                    tensor=f"{short_label(name, 38)} ({index}/{total})",
                )

            file_metadatas.append(
                build_file_metadata(file_path, relative_path, on_tensor=on_tensor)
            )
            progress.advance(task_id)

    return FolderIndex(folder_index=file_metadatas)


def get_remote_chunk_metadata_from_index(folder_index: FolderIndex, remote_file_name: str) -> FileMetadata | None:
    """
        Retrieve file metadata from the index by remote file name.
    :param folder_index:
    :param remote_file_name:
    :return:
    """
    for file_metadata in folder_index.folder_index:
        if file_metadata.file_name == remote_file_name:
            return file_metadata
    return None
