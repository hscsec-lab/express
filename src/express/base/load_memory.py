"""Pre-flight memory checks before loading full weights for inspection."""

from __future__ import annotations

from pathlib import Path

_PROC_MEMINFO = Path("/proc/meminfo")


def estimate_weight_bytes(model_dir: Path) -> int:
    """
    Sum on-disk weight shards (``.safetensors`` and ``*model*.bin``).

    Close to loaded parameter bytes; header / index overhead is negligible.
    """
    total = 0
    for path in model_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        name = path.name.lower()
        if suffix == ".safetensors":
            total += path.stat().st_size
        elif suffix == ".bin" and "model" in name:
            total += path.stat().st_size
    return total


def available_device_memory_bytes(device: str) -> int | None:
    """
    Free bytes on *device* (``cuda`` VRAM or host RAM for ``cpu``).

    Returns ``None`` if the OS free RAM cannot be determined (non-Linux host).
    """
    import torch

    if device == "cuda":
        if not torch.cuda.is_available():
            return 0
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    return _host_mem_available_bytes()


def _host_mem_available_bytes() -> int | None:
    try:
        with _PROC_MEMINFO.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _format_bytes(num: int) -> str:
    if num < 1024:
        return f"{num} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        num_f = num / 1024
        if num_f < 1024:
            return f"{num_f:.2f} {unit}"
        num = int(num_f)
    return f"{num / 1024:.2f} PiB"


def inspection_memory_shortfall(
        model_dir: Path,
        device: str,
        *,
        weight_bytes: int | None = None,
        headroom: float = 1.20,
) -> tuple[int, int, int] | None:
    """
    If full materialized load likely exceeds free memory, return
    ``(weights, needed_with_headroom, available)``; otherwise ``None``.
    """
    weights = weight_bytes if weight_bytes is not None else estimate_weight_bytes(model_dir)
    if weights <= 0:
        return None

    needed = int(weights * headroom)
    available = available_device_memory_bytes(device)
    if available is None:
        return None

    if needed <= available:
        return None
    return weights, needed, available


def _insufficient_memory_message(device: str, weights: int, needed: int, available: int) -> str:
    target = "GPU VRAM" if device == "cuda" else "system RAM"
    hints = [
        f"weights ~{_format_bytes(weights)} (need ~{_format_bytes(needed)} with headroom), "
        f"only ~{_format_bytes(available)} free on {device}.",
    ]
    if device == "cuda":
        hints.append(
            "Free GPU memory, use a larger GPU, or run "
            "`express view ... --device cpu --no-fp --no-er` for structure-only (meta/offload)."
        )
    else:
        hints.append(
            "Use `express view ... --no-fp --no-er` to browse structure without full RAM load "
            "(accelerate meta/disk offload, like older Express), or free RAM / use `--full-load` "
            "only if you accept swap/OOM risk."
        )
    return f"Insufficient {target} to materialize all weights: " + " ".join(hints)


def check_inspection_memory_fits(
        model_dir: Path,
        device: str,
        *,
        weight_bytes: int | None = None,
        headroom: float = 1.20,
) -> None:
    """
    Raise ``RuntimeError`` if free memory on *device* is likely insufficient for a full load.

    *headroom* accounts for allocator overhead (default 20%).
    Skips the check when weight size or host RAM is unknown.
    """
    shortfall = inspection_memory_shortfall(
        model_dir, device, weight_bytes=weight_bytes, headroom=headroom
    )
    if shortfall is None:
        return
    weights, needed, available = shortfall
    raise RuntimeError(_insufficient_memory_message(device, weights, needed, available))
