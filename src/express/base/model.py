import ast
import importlib
import json
import operator
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
from transformers import PreTrainedModel, AutoTokenizer, AutoModel

from express import console
from express.base.data import from_torrent, Torrent, get_torrent
from express.base.file import generate_index, FolderIndex
from rich.pretty import Pretty


SEMVER_PATTERN = r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
MODEL_INDEX_FILE_NAME = os.getenv("MODEL_INDEX_FILE_NAME", 'express-index.json')
METADATA_FILE_NAME = os.getenv("METADATA_FILE_NAME", 'metadata.json')
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def evaluate_model_expression(expr: str, model_map: dict):
    """
    解析并计算模型表达式
    :param expr: 表达式字符串，例如 "(A + B) * 0.5"
    :param model_map: 变量名到 Model 对象的映射
    """
    tree = ast.parse(expr, mode='eval')

    def _eval(node):
        if isinstance(node, ast.BinOp):
            return OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.Num):  # 支持数字缩放
            return node.n
        elif isinstance(node, ast.Constant):  # 兼容新版 Python
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in model_map:
                return model_map[node.id]
            raise ValueError(f"未定义的模型变量: {node.id}")
        else:
            raise TypeError(f"不支持的表达式语法: {type(node)}")

    return _eval(tree.body)


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
        # 确保两个模型都已加载
        model_a = self.model

        sd_a = model_a.state_dict()  # 虽然后面会直接把运算结果覆盖到sd_a，但是sd_a不会进行保存，为了节省资源，我们直接在原模型权重上进行操作
        if isinstance(other, Model):
            # 与模型计算
            sd_b = other.model.state_dict()
            with torch.no_grad():
                for key in tqdm(sd_a.keys(), desc="Model Op"):
                    if key in sd_b:
                        # 确保维度一致
                        if sd_a[key].shape == sd_b[key].shape:
                            # 执行运算：tensor + tensor
                            sd_a[key] = op_func(sd_a[key], sd_b[key].to(sd_a[key].device))

        elif isinstance(other, (int, float)):
            # 与标量计算
            with torch.no_grad():
                for key in sd_a.keys():
                    # 执行运算：tensor + scalar (Torch 会自动处理逐元素运算)
                    sd_a[key] = op_func(sd_a[key], other)

        else:
            raise TypeError(f"Unsupported type: {type(other)}. Must be Model, int, or float.")
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

    def __rmul__(self, other):
        return self.__mul__(other)  # 乘法满足交换律

    def __rtruediv__(self, other):
        # 这是处理 scalar / model 的情况，逻辑稍有不同
        return self._apply_op(other, lambda a, b: b / (a + 1e-12))

    def _load(self, device: str = None, **kwargs) -> PreTrainedModel:
        """
        加载并返回真正的 PreTrainedModel 实例
        等价于return AutoModel.from_pretrained(...)
        """
        if self._model_instance is None:
            console.print(f"Loading weights from {self.path}")
            has_accelerate = importlib.util.find_spec("accelerate") is not None
            if not device:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            load_params = {
                "pretrained_model_name_or_path": self.path
            }
            if has_accelerate:
                load_params["device_map"] = "auto"
                self._model_instance = AutoModel.from_pretrained(**load_params)
            else:
                console.print(f"Accelerate not found. Falling back to single device: {device}")
                self._model_instance = AutoModel.from_pretrained(**load_params).to(device)
        return self._model_instance

    def _save(self, instance: PreTrainedModel, suffix: str = "_output") -> "Model":
        """
        Save Model instance to a new temporary directory and return a new Model object pointing to it.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="model_op_", suffix=suffix))
        console.print(f"Saving model to {temp_dir}")
        instance.save_pretrained(temp_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.path)
        console.print(f"Saving tokenizers to {temp_dir}")
        tokenizer.save_pretrained(temp_dir)
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
