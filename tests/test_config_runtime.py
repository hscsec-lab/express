import json
from pathlib import Path

import express.base.config as cfg
import pytest
from express.base.config import ensure_remote_config, load_config, save_config


def test_load_config_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "config_dir", lambda: tmp_path / "express")
    assert load_config() == {}


def test_load_config_invalid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "config_dir", lambda: tmp_path / "express")
    path = cfg.config_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Failed to read"):
        load_config()


def test_load_config_not_object(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "config_dir", lambda: tmp_path / "express")
    path = cfg.config_path()
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected a JSON object"):
        load_config()


def test_ensure_remote_config_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "config_dir", lambda: tmp_path / "express")
    for key in ("S3_AK", "S3_SK", "S3_ENDPOINT", "S3_BUCKET"):
        monkeypatch.setenv(key, "v")
    assert ensure_remote_config() is None
    cfg.save_config({k: "v" for k in ("S3_AK", "S3_SK", "S3_ENDPOINT", "S3_BUCKET")})
    assert ensure_remote_config() == cfg.config_path()


def test_ensure_remote_config_non_tty(monkeypatch):
    for key in ("S3_AK", "S3_SK", "S3_ENDPOINT", "S3_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cfg, "load_config", lambda: {})
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(EnvironmentError, match="no interactive terminal"):
        ensure_remote_config()
