from __future__ import annotations

import ast
import importlib
import json
import operator
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Any

import rich
from pydantic import BaseModel, field_validator, Field
from pydantic_core.core_schema import ValidationInfo
from rich.pretty import Pretty

from express import console
from express.base.data import from_torrent, Torrent, get_torrent
from express.base.file import generate_index, FolderIndex

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PretrainedConfig

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
    Parse and calculate model expression
    :param expr: Expression string, e.g., "(A + B) * 0.5"
    :param model_map: Mapping from variable name to Model object
    """
    tree = ast.parse(expr, mode='eval')

    def _eval(node):
        if isinstance(node, ast.BinOp):
            return OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.Num):  # Support numeric scaling
            return node.n
        elif isinstance(node, ast.Constant):  # Compatible with newer Python versions
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in model_map:
                return model_map[node.id]
            raise ValueError(f"Undefined model variable: {node.id}")
        else:
            raise TypeError(f"Unsupported expression syntax: {type(node)}")

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
        :param path: Model folder path
        """
        if not path.exists():
            console.print(f"Created model {path}")
            path.mkdir()
        self.metadata_file_name = METADATA_FILE_NAME
        self.path = path
        self.folder_index: FolderIndex = self.write_index_file()  # Overwrite to the latest index each time it is called
        self._model_instance: Optional[Any] = None  # Lazy load model instance

        assert self.path.is_dir(), f"{self.path} must be directory."

    @staticmethod
    def _compute_tensors(sd_a, sd_b, op_func):
        import torch

        with torch.no_grad():
            common_keys = set(sd_a.keys()) & set(sd_b.keys())
            for key in common_keys:
                if sd_a[key].shape == sd_b[key].shape:
                    sd_a[key] = op_func(sd_a[key], sd_b[key].to(sd_a[key].device))
        return sd_a

    def _apply_op(self, other, op_func) -> "Model":
        import torch

        model_a = self.model
        sd_a = model_a.state_dict()

        if isinstance(other, Model):
            sd_a = self._compute_tensors(sd_a, other.model.state_dict(), op_func)
        elif isinstance(other, (int, float)):
            with torch.no_grad():
                sd_a = {k: op_func(v, other) for k, v in sd_a.items()}
        else:
            raise TypeError(f"Unsupported type: {type(other)}")

        model_a.load_state_dict(sd_a)
        return self._save(model_a)

    def __add__(self, other):
        """Addition: model1 + model2"""
        return self._apply_op(other, lambda a, b: a + b)

    def __sub__(self, other):
        """Subtraction: model1 - model2"""
        return self._apply_op(other, lambda a, b: a - b)

    def __mul__(self, other):
        """Multiplication: model1 * model2 (Hadamard product)"""
        return self._apply_op(other, lambda a, b: a * b)

    def __truediv__(self, other):
        """Division: model1 / model2"""
        # Note: Division needs to prevent division by zero
        return self._apply_op(other, lambda a, b: a / (b + 1e-12))

    def __rmul__(self, other):
        return self.__mul__(other)  # Multiplication is commutative

    def __rtruediv__(self, other):
        # This handles the case of scalar / model, with slightly different logic
        return self._apply_op(other, lambda a, b: b / (a + 1e-12))

    def _load(self, device: Optional[str] = None, **kwargs) -> PreTrainedModel:
        """
        Main entry point for loading the model with caching mechanism and automated dispatching.
        """
        from transformers import AutoConfig

        if self._model_instance is not None:
            return self._model_instance

        config = AutoConfig.from_pretrained(self.path, trust_remote_code=True)

        load_class = self._determine_load_class(config)
        load_params = self._prepare_load_params(config, **kwargs)

        self._model_instance = self._execute_model_loading(
            load_class,
            load_params,
            device
        )

        return self._model_instance

    def _determine_load_class(self, config: PretrainedConfig) -> type:
        """
        Determines the appropriate AutoModel class based on the provided configuration mapping.
        """
        from transformers import (
            AutoModelForAudioFrameClassification,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForSeq2SeqLM,
        )

        dispatch_map = [
            (AutoModelForImageTextToText, "vision-language"),
            (AutoModelForSeq2SeqLM, "audio-seq2seq/omni"),
            (AutoModelForAudioFrameClassification, "audio-classification"),
            (AutoModelForCausalLM, "causal-llm")
        ]

        for cls, label in dispatch_map:
            if type(config) in cls._model_mapping.keys():
                return cls
        return AutoModelForCausalLM

    def _prepare_load_params(self, config: PretrainedConfig, **kwargs) -> dict:
        """
        Constructs the standard parameter dictionary for the from_pretrained call.
        """
        return {
            "pretrained_model_name_or_path": self.path,
            "config": config,
            "trust_remote_code": True,
            "torch_dtype": "auto",
            **kwargs
        }

    def _execute_model_loading(
            self,
            load_class: type,
            load_params: dict,
            device: Optional[str]
    ) -> PreTrainedModel:
        """
        Executes the actual model loading logic, handling accelerate integration or manual device placement.
        """
        import torch

        offload_folder = os.getenv("OFFLOAD_FOLDER", "/tmp/.cache")

        if importlib.util.find_spec("accelerate"):
            load_params.setdefault("device_map", "auto")
            return load_class.from_pretrained(
                **load_params,
                offload_folder=offload_folder
            )

        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return load_class.from_pretrained(
            **load_params,
            offload_folder=offload_folder
        ).to(target_device)

    def _save(self, instance: PreTrainedModel, suffix: str = "_output") -> "Model":
        """
        Save Model instance to a new temporary directory and return a new Model object pointing to it.
        """
        from transformers import AutoTokenizer

        temp_dir = Path(tempfile.mkdtemp(prefix="model_op_", suffix=suffix))
        console.print(f"Saving model to {temp_dir}")
        instance.save_pretrained(temp_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.path)
        console.print(f"Saving tokenizers to {temp_dir}")
        tokenizer.save_pretrained(temp_dir)
        return Model(temp_dir)

    @property
    def model(self) -> PreTrainedModel:
        """Conveniently access the model instance via model.model"""
        return self._load()

    def write_index_file(self) -> FolderIndex:
        console.print(f"Writing index file to {self.path}")
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
        Get the metadata of the model
        :return:
        """
        metadata_path = self.path / self.metadata_file_name
        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return Metadata(**data)

    def create_metadata(self, authors="bob", emails="human@human.com", version="0.1.0", tags=None, name="default"):
        """
        Create metadata for the model
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
