#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p docker
uv export --no-dev --no-emit-project --no-hashes -o docker/requirements.runtime.txt
echo "Wrote docker/requirements.runtime.txt ($(wc -l < docker/requirements.runtime.txt) lines)"
