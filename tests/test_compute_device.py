import pytest

from express.base.compute_device import resolve_compute_device, resolve_svd_flags


def test_resolve_auto_cpu_when_no_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False, raising=False)
    import torch

    monkeypatch.setattr(torch, "cuda", torch.cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_compute_device(None) == "cpu"
    assert resolve_compute_device("auto") == "cpu"


def test_resolve_auto_cuda_when_available(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_compute_device("auto") == "cuda"


def test_resolve_explicit_cpu():
    assert resolve_compute_device("cpu") == "cpu"


def test_resolve_explicit_cuda_available(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_compute_device("cuda") == "cuda"


def test_resolve_explicit_cuda_unavailable(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_compute_device("cuda")


def test_resolve_invalid():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_compute_device("tpu")


def test_resolve_svd_defaults():
    assert resolve_svd_flags("cuda", None, None) == (True, True)
    assert resolve_svd_flags("cpu", None, None) == (False, False)


def test_resolve_svd_overrides():
    assert resolve_svd_flags("cpu", True, False) == (True, False)
    assert resolve_svd_flags("cuda", False, True) == (False, True)
