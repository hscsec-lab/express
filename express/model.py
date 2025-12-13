import json
import re
import zlib
from pathlib import Path
from typing import Optional, List

import rich
from pydantic import BaseModel, field_validator, Field
from pydantic_core.core_schema import ValidationInfo

from express import console
from express.data import from_torrent, Torrent, get_torrent
from express.file import generate_index, FolderIndex
from rich.pretty import pprint, Pretty

SEMVER_PATTERN = r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"

class Metadata(BaseModel):
    authors: Optional[str] = None
    emails: Optional[str] = None
    version: Optional[str] = Field(
        None,
        description="Version string in Semantic Versioning 2.0.0 format (e.g., '1.2.3', '1.0.0-rc.1+build.456')"
    )
    tags: Optional[List[str]] = None
    name: str = None

    @field_validator("version", mode="after")
    @classmethod
    def validate_semver(cls, v: str | None, info: ValidationInfo) -> str | None:
        if v is not None:
            if not re.fullmatch(SEMVER_PATTERN, v):
                raise ValueError(f"Version '{v}' is not a valid SemVer 2.0.0 string.")
        return v

    def get_torrent(self) -> Torrent:
        return get_torrent(self)

    @staticmethod
    def from_torrent(torrent: Torrent) -> "Metadata":
        return from_torrent(torrent,Metadata)

class Model:
    def __init__(self, path: Path):
        """
        Init
        :param path: 模型文件夹路径
        """
        if not path.exists():
            console.print(f"Created model {path}")
            path.mkdir()
        self.metadata_file_name = 'metadata.json'
        self.path = path
        self.index_file_name = "index.json"
        self.folder_index:FolderIndex = self.create_index_file() # 每次调用覆盖到最新的索引

        assert self.path.is_dir(), f"{self.path} must be directory."

    def create_index_file(self) -> FolderIndex:
        folder_index: FolderIndex = generate_index(self.path)
        with (self.path / self.index_file_name).open('w',encoding='utf-8') as f:
            json.dump(
                folder_index.model_dump(exclude_none=False),
                f,
                ensure_ascii=False,
                indent=2
            )
            return folder_index

    def is_index_file_exists(self):
        return (self.path / self.index_file_name).exists()

    def is_metadata_file_exists(self):
        return (self.path / self.metadata_file_name).exists()

    def get_metadata(self) -> Metadata:
        """
        获取模型的metadata
        :return:
        """
        metadata_path = self.path / self.metadata_file_name
        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return Metadata(**data)

    def create_metadata(self, name="default"):
        """
        创建模型的metadata
        :return:
        """
        metadata_path = self.path / self.metadata_file_name
        metadata = Metadata(
            authors="example_authors",
            emails="human@human.com",
            version="0.0.0",
            tags=["example_tag"],
            name=name
        )
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(
                metadata.model_dump(exclude_none=False),
                f,
                ensure_ascii=False,
                indent=2
            )

    def remove_metadata(self):
        metadata_path = self.path / self.metadata_file_name
        metadata_path.unlink()

def get_metadata(torrent: Torrent):
    metadata:Metadata = from_torrent(torrent,Metadata)
    rich.print(Pretty(metadata.model_dump(), expand_all=True, indent_guides=False))
