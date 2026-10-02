"""Fast model structure from safetensors headers only (no weight I/O, no transformers load)."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

from tqdm import tqdm

from express.base.storage.safetensors import _is_tensor_entry


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) < 8:
            raise ValueError(f"not a safetensors file: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > 32 * 1024 * 1024:
            raise ValueError(f"safetensors header too large: {path}")
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header: {path}")
    return header


def _shard_paths(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map") or {}
        paths = []
        for rel in sorted(set(weight_map.values())):
            candidate = model_dir / rel
            if candidate.is_file():
                paths.append(candidate)
        if paths:
            return paths
    return sorted(p for p in model_dir.rglob("*.safetensors") if p.is_file())


def scan_model_structure(model_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Build view-layer analysis rows from safetensors JSON headers only.

    Same keys as ``pre_analyze_model`` output; FP/ER left empty.
    """
    shards = _shard_paths(model_dir)
    if not shards:
        raise RuntimeError(
            f"No .safetensors weights under {model_dir}. "
            "Use --full-load to inspect via transformers (slow, reads all weights)."
        )

    rows: dict[str, dict[str, Any]] = {}
    for shard in tqdm(shards, desc="Reading headers"):
        header = _read_safetensors_header(shard)
        for name, entry in header.items():
            if not _is_tensor_entry(entry):
                continue
            if name in rows:
                raise ValueError(f"duplicate tensor name {name!r} in {shard}")
            shape = [int(x) for x in entry["shape"]]
            params = math.prod(shape)
            rows[name] = {
                "params": params,
                "shape": shape,
                "fp": "",
                "er": 0.0,
            }
    if not rows:
        raise RuntimeError(f"No tensors found in safetensors under {model_dir}")
    return rows
