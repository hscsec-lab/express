import json
import re
import zlib
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, field_validator, Field
from pydantic_core.core_schema import ValidationInfo

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

    def get_digest(self) -> str:
        return zlib.compress(self.model_dump_json().encode()).hex()

    @staticmethod
    def from_digest(digest: str) -> "Metadata":
        compressed = bytes.fromhex(digest)
        json_bytes = zlib.decompress(compressed)
        json_str = json_bytes.decode()
        return Metadata.model_validate_json(json_str)

class Model:
    def __init__(self, path: Path):
        """
        Init
        :param path: 模型文件夹路径
        """
        self.metadata_file_name = 'metadata.json'
        self.path = path

        assert self.path.is_dir(), f"{self.path} must be directory."

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
        metadata.model_dump_json()
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
