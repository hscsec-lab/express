from pathlib import Path
from typing import List

from pydantic import BaseModel, field_serializer, field_validator
from simple_file_checksum import get_checksum


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

class FolderIndex(BaseModel):
        folder_index: List[FileMetadata] = []

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
                    file_checksum_sha256=get_checksum(file_path),
                    file_relative_path=relative_path
                )
            )
    return FolderIndex(folder_index=file_metadatas)

if __name__ == '__main__':
    print(generate_index(Path('../express')))