#!/usr/bin/env bash
set -euo pipefail

START=$(date +%s)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MARKER=".venv/.ci-deps-hash"
FINGERPRINT="$(sha256sum pyproject.toml uv.lock 2>/dev/null | sha256sum | awk '{print $1}')"

export PIP_DISABLE_PIP_VERSION_CHECK=1

if [[ -x .venv/bin/python && -f "$MARKER" && "$(cat "$MARKER")" == "$FINGERPRINT" ]]; then
  echo "python setup: reuse cached venv"
else
  echo "python setup: install deps"
  rm -rf .venv
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install --no-cache-dir -e ".[dev]"
  mkdir -p "$(dirname "$MARKER")"
  echo "$FINGERPRINT" > "$MARKER"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -c "import express, pytest, safetensors, numpy"
END=$(date +%s)
echo "python setup elapsed: $((END - START))s"
