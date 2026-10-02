"""Semantic equivalence checks for .safetensors files (independent of on-disk byte layout)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.text import Text

from express import console
from express.base.storage.safetensors import SafetensorsHandler

WHOLE_FILE_HASH_HINT = (
    "Whole-file SHA256 / fast-checksum may differ after Express sync; "
    "tensor data and __metadata__ should still match. "
    "Compare with: [cyan]express cmp-safetensors[/cyan] LEFT RIGHT"
)


@dataclass(frozen=True)
class SafetensorsEquivalenceResult:
    equivalent: bool
    differences: tuple[str, ...]
    bytes_identical: bool


def _load_view(path: Path) -> tuple[dict[str, str] | None, dict[str, Any]]:
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        metadata = handle.metadata()
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    return metadata, tensors


def compare_safetensors_paths(left: Path, right: Path) -> SafetensorsEquivalenceResult:
    """
    Return whether two files are equivalent per the safetensors library (tensors + metadata).

    Does not require byte-identical files.
    """
    left = Path(left)
    right = Path(right)
    diffs: list[str] = []

    if not SafetensorsHandler.matches(left):
        return SafetensorsEquivalenceResult(False, (f"Not a safetensors file: {left}",), False)
    if not SafetensorsHandler.matches(right):
        return SafetensorsEquivalenceResult(False, (f"Not a safetensors file: {right}",), False)

    bytes_identical = left.read_bytes() == right.read_bytes()
    meta_l, tensors_l = _load_view(left)
    meta_r, tensors_r = _load_view(right)

    if meta_l != meta_r:
        diffs.append(f"metadata differs: {meta_l!r} vs {meta_r!r}")
    keys_l = set(tensors_l)
    keys_r = set(tensors_r)
    if keys_l != keys_r:
        only_l = sorted(keys_l - keys_r)
        only_r = sorted(keys_r - keys_l)
        if only_l:
            diffs.append(f"tensors only in {left.name}: {only_l}")
        if only_r:
            diffs.append(f"tensors only in {right.name}: {only_r}")
    for name in sorted(keys_l & keys_r):
        import numpy as np

        a, b = tensors_l[name], tensors_r[name]
        if a.dtype != b.dtype:
            diffs.append(f"{name}: dtype {a.dtype} != {b.dtype}")
            continue
        if a.shape != b.shape:
            diffs.append(f"{name}: shape {a.shape} != {b.shape}")
            continue
        if not np.array_equal(a, b):
            diffs.append(f"{name}: tensor values differ")

    return SafetensorsEquivalenceResult(
        equivalent=not diffs,
        differences=tuple(diffs),
        bytes_identical=bytes_identical,
    )


def print_whole_file_hash_hint(*, path: Path | None = None) -> None:
    """One-line user hint that byte-level hashes may differ for safetensors."""
    label = path.name if path is not None else "safetensors"
    console.print(
        Text.from_markup(
            f"[dim]ℹ {label}: {WHOLE_FILE_HASH_HINT}[/dim]",
            overflow="fold",
        )
    )
