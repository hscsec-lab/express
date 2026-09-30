#!/usr/bin/env python3
"""Measure push/pull timing for safetensors tensor-level sync (local dev / S3)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from express.base.client import Remote
from express.base.model import Model
from express.base.remote import pull_model_with_index, push
from express.base.storage.registry import build_file_metadata


def _sec(start: float) -> float:
    return round(time.perf_counter() - start, 2)


def main() -> None:
    work = ROOT / ".benchmark-run"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    model_dir = work / "tiny-safetensors-model"
    model_dir.mkdir()
    save_file(
        {
            "layer.weight": np.random.randn(8, 8).astype(np.float32) * 0.02,
            "layer.bias": np.zeros(8, dtype=np.float32),
        },
        model_dir / "model.safetensors",
    )
    (model_dir / "config.json").write_text('{"model_type":"benchmark-tiny"}', encoding="utf-8")

    model = Model(model_dir)
    model.create_metadata(
        name="benchmark-tiny-safetensors",
        version="0.0.1",
        authors="benchmark",
        tags=["benchmark", "safetensors"],
    )
    model.write_index_file()

    remote = Remote()
    results: list[tuple[str, float, str]] = []

    t0 = time.perf_counter()
    push(model, remote)
    results.append(("首次 push（全量 tensor blobs + 索引）", _sec(t0), "冷启动上传"))

    index = json.loads((model_dir / "express-index.json").read_text(encoding="utf-8"))
    pull_dir = work / "pull-1"
    pull_dir.mkdir()
    t0 = time.perf_counter()
    pull_model_with_index(index, remote, pull_dir, force=True)
    results.append(("首次 pull（还原 safetensors）", _sec(t0), "下载全部 tensor blobs"))

    before = build_file_metadata(model_dir / "model.safetensors", Path("model.safetensors"))
    before_parts = {p["name"]: p["content_hash"] for p in before.unit_manifest["parts"]}

    save_file(
        {
            "layer.weight": np.ones((8, 8), dtype=np.float32),
            "layer.bias": np.zeros(8, dtype=np.float32),
        },
        model_dir / "model.safetensors",
    )
    model.write_index_file()
    after = build_file_metadata(model_dir / "model.safetensors", Path("model.safetensors"))
    after_parts = {p["name"]: p["content_hash"] for p in after.unit_manifest["parts"]}

    changed = [n for n in before_parts if before_parts[n] != after_parts[n]]
    unchanged = [n for n in before_parts if before_parts[n] == after_parts[n]]

    t0 = time.perf_counter()
    push(model, remote)
    results.append(("二次 push（仅变更 tensor）", _sec(t0), f"改 {changed}；跳过 {unchanged}"))

    index2 = json.loads((model_dir / "express-index.json").read_text(encoding="utf-8"))
    pull_dir2 = work / "pull-2"
    pull_dir2.mkdir()
    t0 = time.perf_counter()
    pull_model_with_index(index2, remote, pull_dir2, force=True)
    results.append(("二次 pull", _sec(t0), "按新 manifest 拉取"))

    print("MODEL_DIR", model_dir)
    print("TENSORS", list(before_parts.keys()))
    print("CHANGED", changed)
    print("UNCHANGED", unchanged)
    print("TIMINGS_JSON", json.dumps(results, ensure_ascii=False))
    for label, sec, note in results:
        print(f"{label}\t{sec}s\t{note}")


if __name__ == "__main__":
    main()
