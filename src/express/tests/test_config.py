import json
import os
from pathlib import Path

import express.base.config as cfg
from express.base.config import (
    BACK_TOKENS,
    GoBack,
    apply_config_to_env,
    default_local_workdir,
    load_config,
    missing_required_keys,
    save_config,
)


def test_save_and_load_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "config_dir", lambda: tmp_path / "express")
    payload = {
        "S3_AK": "ak",
        "S3_SK": "sk",
        "S3_ENDPOINT": "https://example.com",
        "S3_BUCKET": "bucket",
        "LOCAL_WORKDIR": str(tmp_path / "models"),
    }
    path = save_config(payload)
    assert path.exists()
    assert oct(path.stat().st_mode)[-3:] == "600"
    loaded = load_config()
    assert loaded["S3_BUCKET"] == "bucket"
    assert loaded["LOCAL_WORKDIR"] == str(tmp_path / "models")


def test_env_overrides_config(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "config_dir", lambda: tmp_path / "express")
    save_config({
        "S3_AK": "ak",
        "S3_SK": "sk",
        "S3_ENDPOINT": "https://example.com",
        "S3_BUCKET": "from-file",
        "LOCAL_WORKDIR": str(tmp_path / "models"),
    })
    for key in ("S3_AK", "S3_SK", "S3_ENDPOINT", "S3_BUCKET", "LOCAL_WORKDIR"):
        monkeypatch.delenv(key, raising=False)
    apply_config_to_env(load_config())
    assert os.getenv("S3_BUCKET") == "from-file"
    monkeypatch.setenv("S3_BUCKET", "from-env")
    apply_config_to_env(load_config())
    assert os.getenv("S3_BUCKET") == "from-env"


def test_missing_required_keys(monkeypatch):
    for key in ("S3_AK", "S3_SK", "S3_ENDPOINT", "S3_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    assert set(missing_required_keys()) == {"S3_AK", "S3_SK", "S3_ENDPOINT", "S3_BUCKET"}


def test_back_tokens_and_default_workdir():
    assert "-" in BACK_TOKENS
    assert ":back" in BACK_TOKENS
    assert isinstance(GoBack(), GoBack)
    assert Path(default_local_workdir()).name == "models"
