from pathlib import Path
from unittest.mock import MagicMock

import pytest

from express.base.load_memory import (
    _format_bytes,
    available_device_memory_bytes,
    check_inspection_memory_fits,
    estimate_weight_bytes,
)


def test_estimate_weight_bytes(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "pytorch_model.bin").write_bytes(b"y" * 50)
    (tmp_path / "optimizer.bin").write_bytes(b"z" * 999)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.safetensors").write_bytes(b"n" * 10)
    (tmp_path / "empty_dir").mkdir()
    assert estimate_weight_bytes(tmp_path) == 160


def test_format_bytes():
    assert _format_bytes(100) == "100 B"
    assert "KiB" in _format_bytes(2048)
    assert "GiB" in _format_bytes(2 * 1024**3)
    assert "PiB" in _format_bytes(1024**5)


def test_check_fits_cuda(monkeypatch, tmp_path):
    (tmp_path / "w.safetensors").write_bytes(b"a" * 1000)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (500, 1000))
    with pytest.raises(RuntimeError, match="Insufficient GPU VRAM"):
        check_inspection_memory_fits(tmp_path, "cuda", weight_bytes=1000, headroom=1.0)


def test_check_passes_when_enough(monkeypatch, tmp_path):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (10_000_000, 20_000_000))
    check_inspection_memory_fits(tmp_path, "cuda", weight_bytes=1000, headroom=1.2)


def test_host_mem_available(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "express.base.load_memory._host_mem_available_bytes",
        lambda: 8 * 1024**3,
    )
    assert available_device_memory_bytes("cpu") == 8 * 1024**3
    check_inspection_memory_fits(tmp_path, "cpu", weight_bytes=1024, headroom=1.0)


def test_skip_when_no_weights(tmp_path):
    check_inspection_memory_fits(tmp_path, "cuda", weight_bytes=0)


def test_host_mem_from_proc():
    from express.base.load_memory import _host_mem_available_bytes

    if not Path("/proc/meminfo").is_file():
        pytest.skip("Linux /proc/meminfo only")
    assert (_host_mem_available_bytes() or 0) > 0


def test_host_mem_open_error(tmp_path, monkeypatch):
    from express.base import load_memory

    monkeypatch.setattr(load_memory, "_PROC_MEMINFO", tmp_path / "missing-meminfo")
    assert load_memory._host_mem_available_bytes() is None


def test_host_mem_missing_memavailable(tmp_path, monkeypatch):
    from express.base import load_memory

    fake = tmp_path / "meminfo"
    fake.write_text("MemTotal:       1000 kB\n", encoding="utf-8")
    monkeypatch.setattr(load_memory, "_PROC_MEMINFO", fake)
    assert load_memory._host_mem_available_bytes() is None


def test_cuda_unavailable_reports_zero(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert available_device_memory_bytes("cuda") == 0


def test_cpu_insufficient_ram(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "express.base.load_memory._host_mem_available_bytes",
        lambda: 100,
    )
    with pytest.raises(RuntimeError, match="system RAM"):
        check_inspection_memory_fits(tmp_path, "cpu", weight_bytes=200, headroom=1.0)


def test_skip_when_host_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "express.base.load_memory._host_mem_available_bytes",
        lambda: None,
    )
    (tmp_path / "m.safetensors").write_bytes(b"z" * 10_000_000)
    check_inspection_memory_fits(tmp_path, "cpu")
