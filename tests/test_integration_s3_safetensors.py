"""Optional live S3 test: pytest --run-integration tests/test_integration_s3_safetensors.py"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from express.base.client import Remote
from express.base.model import Model
from express.base.remote import pull_model_with_index, push
from express.base.s3_transfer import download_file_with_resume, partial_path_for, upload_file_with_retries
from express.base.storage.registry import build_file_metadata
from express.base.storage.safetensors import SafetensorsHandler

pytestmark = pytest.mark.integration


def _require_s3_env() -> None:
    for key in ("S3_AK", "S3_SK", "S3_ENDPOINT", "S3_BUCKET"):
        if not os.getenv(key):
            pytest.skip(f"missing {key} for integration test")


@pytest.fixture
def tiny_model_dir(tmp_path) -> Path:
    model_dir = tmp_path / "tiny-safetensors-model"
    model_dir.mkdir()
    save_file(
        {
            "layer0.weight": np.random.randn(4, 4).astype(np.float32),
            "layer0.bias": np.zeros(4, dtype=np.float32),
        },
        model_dir / "model.safetensors",
    )
    (model_dir / "config.json").write_text('{"model_type":"demo"}', encoding="utf-8")
    return model_dir


def test_push_pull_and_partial_tensor_update(tiny_model_dir, tmp_path):
    _require_s3_env()
    model = Model(tiny_model_dir)
    model.create_metadata(name="tiny-safetensors-model", version="0.0.1", tags=["integration"])
    model.write_index_file()

    remote = Remote()
    push(model, remote)

    pull_dir = tmp_path / "pulled"
    pull_dir.mkdir()
    index = __import__("json").loads((tiny_model_dir / "express-index.json").read_text(encoding="utf-8"))
    pull_model_with_index(index, remote, pull_dir, force=True)
    pulled_weights = pull_dir / "model.safetensors"
    assert pulled_weights.exists()
    source_meta = build_file_metadata(tiny_model_dir / "model.safetensors", Path("model.safetensors"))
    handler = SafetensorsHandler()
    assert handler.verify_local(pulled_weights, source_meta.file_checksum_sha256, source_meta.unit_manifest)

    before_meta = build_file_metadata(tiny_model_dir / "model.safetensors", Path("model.safetensors"))
    tensors = {"layer0.weight": np.ones((4, 4), dtype=np.float32), "layer0.bias": np.zeros(4, dtype=np.float32)}
    save_file(tensors, tiny_model_dir / "model.safetensors")
    model.write_index_file()
    after_meta = build_file_metadata(tiny_model_dir / "model.safetensors", Path("model.safetensors"))

    before_parts = {p["name"]: p["content_hash"] for p in before_meta.unit_manifest["parts"]}
    after_parts = {p["name"]: p["content_hash"] for p in after_meta.unit_manifest["parts"]}
    assert before_parts["layer0.bias"] == after_parts["layer0.bias"]
    assert before_parts["layer0.weight"] != after_parts["layer0.weight"]

    push(model, remote)
    pull_dir2 = tmp_path / "pulled2"
    pull_dir2.mkdir()
    index2 = __import__("json").loads((tiny_model_dir / "express-index.json").read_text(encoding="utf-8"))
    pull_model_with_index(index2, remote, pull_dir2, force=True)
    updated_meta = build_file_metadata(tiny_model_dir / "model.safetensors", Path("model.safetensors"))
    assert handler.verify_local(
        pull_dir2 / "model.safetensors",
        updated_meta.file_checksum_sha256,
        updated_meta.unit_manifest,
    )


def test_upload_download_resume_roundtrip(tmp_path):
    """Upload a blob, then finish a download from a local partial via Range."""
    _require_s3_env()
    remote = Remote()
    payload = b"express-resume-" + os.urandom(256 * 1024)
    src = tmp_path / "src.bin"
    src.write_bytes(payload)
    key = f"express-it-resume-{uuid.uuid4().hex}"
    try:
        upload_file_with_retries(
            remote.s3_client,
            src,
            remote.s3_bucket,
            key,
            expected_size=len(payload),
        )
        dest = tmp_path / "dest.bin"
        partial_path_for(dest).write_bytes(payload[:4096])
        download_file_with_resume(
            remote.s3_client,
            remote.s3_bucket,
            key,
            dest,
            total_bytes=len(payload),
        )
        assert dest.read_bytes() == payload
    finally:
        remote.s3_client.delete_object(Bucket=remote.s3_bucket, Key=key)
