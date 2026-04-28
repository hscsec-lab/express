import copy
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, List

import rich
import torch
from pydantic import BaseModel, field_validator, Field
from pydantic_core.core_schema import ValidationInfo
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel

from express import console
from express.base.data import from_torrent, Torrent, get_torrent
from express.base.file import generate_index, FolderIndex
from rich.pretty import Pretty

from express.functions.view_model import view_model

SEMVER_PATTERN = r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
MODEL_INDEX_FILE_NAME = os.getenv("MODEL_INDEX_FILE_NAME", 'express-index.json')
METADATA_FILE_NAME = os.getenv("METADATA_FILE_NAME", 'metadata.json')


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
        return from_torrent(torrent, Metadata)


class Model:
    def __init__(self, path: Path):
        """
        Init
        :param path: 模型文件夹路径
        """
        if not path.exists():
            console.print(f"Created model {path}")
            path.mkdir()
        self.metadata_file_name = METADATA_FILE_NAME
        self.path = path
        self.folder_index: FolderIndex = self.write_index_file()  # 每次调用覆盖到最新的索引
        self._model_instance: Optional[PreTrainedModel] = None  # 延迟加载模型实例

        assert self.path.is_dir(), f"{self.path} must be directory."

    def _apply_op(self, other, op_func) -> "Model":
        """
        通用操作函数：遍历两个模型的 state_dict 并执行 op_func 并将结果赋值到第一个模型中
        """
        if not isinstance(other, Model):
            raise TypeError(f"Operand must be of type Model, not {type(other)}")

        # 确保两个模型都已加载
        model_a = self.model
        model_b = other.model

        # 为了不破坏原模型，通常我们会 copy 一个新模型返回（注意：这会消耗双倍显存/内存）
        sd_a = model_a.state_dict()  # 虽然后面会直接把运算结果覆盖到sd_a，但是sd_a不会进行保存，为了节省资源，我们直接在原模型权重上进行操作
        sd_b = model_b.state_dict()

        assert len(sd_a.keys()) == len(
            sd_b.keys()), "Models have different number of parameters, cannot apply operation."

        keys = list(sd_a.keys())
        total_keys = len(keys)

        with torch.no_grad():
            for key in tqdm(keys, desc=f"Applying operation", unit="param", total=total_keys):
                if key in sd_b:
                    tensor_a = sd_a[key]
                    tensor_b = sd_b[key]
                    if tensor_a.shape == tensor_b.shape:
                        tensor_b_on_a_device = tensor_b.to(tensor_a.device)
                        new_tensor = op_func(tensor_a, tensor_b_on_a_device)
                        sd_a[key] = new_tensor
                    else:
                        console.print(f"[yellow]Warning:[/yellow] Shape mismatch for {key}, skipping.")
                else:
                    console.print(f"[yellow]Warning:[/yellow] Key {key} not found in the second model.")

        model_a.load_state_dict(sd_a)
        return self._save(model_a)

    def __add__(self, other):
        """加法: model1 + model2"""
        return self._apply_op(other, lambda a, b: a + b)

    def __sub__(self, other):
        """减法: model1 - model2"""
        return self._apply_op(other, lambda a, b: a - b)

    def __mul__(self, other):
        """乘法: model1 * model2 (Hadamard product)"""
        return self._apply_op(other, lambda a, b: a * b)

    def __truediv__(self, other):
        """除法: model1 / model2"""
        # 注意：除法需要防止除以 0
        return self._apply_op(other, lambda a, b: a / (b + 1e-12))

    def _load(self, **kwargs) -> PreTrainedModel:
        """
        加载并返回真正的 PreTrainedModel 实例
        等价于return AutoModelForCausalLM.from_pretrained(...)
        """
        if self._model_instance is None:
            console.print(f"Loading weights from {self.path}")
            # 默认使用 auto 映射设备，你可以根据需求调整
            self._model_instance = AutoModelForCausalLM.from_pretrained(
                self.path
            )
        return self._model_instance

    def _save(self, instance: PreTrainedModel, suffix: str = "_output") -> "Model":
        """
        Save Model instance to a new temporary directory and return a new Model object pointing to it.
        """
        # 创建临时目录
        console.print(f"Saving model to {self.path}")
        temp_dir = Path(tempfile.mkdtemp(prefix="model_op_", suffix=suffix))
        instance.save_pretrained(temp_dir)
        return Model(temp_dir)

    @property
    def model(self) -> PreTrainedModel:
        """方便通过 model.model 直接获取实例"""
        return self._load()

    def write_index_file(self) -> FolderIndex:
        folder_index: FolderIndex = generate_index(self.path)
        with (self.path / MODEL_INDEX_FILE_NAME).open('w', encoding='utf-8') as f:
            json.dump(
                folder_index.model_dump(exclude_none=False),
                f,
                ensure_ascii=False,
                indent=2
            )
            return folder_index

    def is_index_file_exists(self):
        return (self.path / MODEL_INDEX_FILE_NAME).exists()

    def remove_index_file(self):
        index_path = self.path / MODEL_INDEX_FILE_NAME
        if index_path.exists():
            index_path.unlink()

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

    def create_metadata(self, authors="bob", emails="human@human.com", version="0.1.0", tags=None, name="default"):
        """
        创建模型的metadata
        :return:
        """
        metadata_path = self.path / self.metadata_file_name
        metadata = Metadata(
            authors=authors,
            emails=emails,
            version=version,
            tags=tags,
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
        if metadata_path.exists():
            metadata_path.unlink()

    def view_gui(self):
        view_model(self.model, self.path.__str__())


def get_metadata(torrent: Torrent) -> Metadata:
    metadata: Metadata = from_torrent(torrent, Metadata)
    return metadata


def _info(torrent: Torrent, fields: Optional[List[str]] = None):
    data = get_metadata(torrent).model_dump()
    if fields:
        data = {k: v for k, v in data.items() if k in fields}
        if len(data) == 1:
            rich.print(next(iter(data.values())))
            return
    rich.print(Pretty(data, expand_all=True, indent_guides=False))
