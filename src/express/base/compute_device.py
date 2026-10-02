"""Compute device selection for model inspection and analysis helpers."""

from __future__ import annotations

from typing import Optional


def resolve_compute_device(choice: Optional[str] = None) -> str:
    """
    Resolve user device choice to ``cpu`` or ``cuda``.

    * ``None`` / ``auto`` — CUDA when available, otherwise CPU.
    * ``cpu`` / ``cuda`` — explicit placement (raises if CUDA unavailable).
    """
    import torch

    normalized = (choice or "auto").strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested (--device cuda) but no GPU is available. "
                "Use --device cpu or --device auto."
            )
        return "cuda"
    raise ValueError(f"Unknown device {choice!r}; use cpu, cuda, or auto.")


def resolve_svd_flags(
        device: str,
        calc_fp: Optional[bool],
        calc_er: Optional[bool],
) -> tuple[bool, bool]:
    """
    Default FP/ER to on when analysis runs on CUDA; off on CPU unless overridden.
    """
    use_gpu = device == "cuda"
    fp = use_gpu if calc_fp is None else calc_fp
    er = use_gpu if calc_er is None else calc_er
    return fp, er
