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


def check_inspection_memory_fits(
        model_dir: Path,
        device: str,
        *,
        weight_bytes: int | None = None,
        headroom: float = 1.20,
) -> None:
    """
    Raise ``RuntimeError`` if free memory on *device* is likely insufficient.

    *headroom* accounts for allocator overhead (default 20%).
    Skips the check when weight size or host RAM is unknown.
    """
    weights = weight_bytes if weight_bytes is not None else estimate_weight_bytes(model_dir)
    if weights <= 0:
        return

    needed = int(weights * headroom)
    available = available_device_memory_bytes(device)
    if available is None:
        return

    if needed <= available:
        return

    target = "GPU VRAM" if device == "cuda" else "system RAM"
    raise RuntimeError(
        f"Insufficient {target} to load this model for inspection: "
        f"weights ~{_format_bytes(weights)} (need ~{_format_bytes(needed)} with headroom), "
        f"but only ~{_format_bytes(available)} free on {device}. "
        f"Try `express view ... --device cpu`, free memory, or use a machine with more {target}."
    )
