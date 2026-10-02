#!/usr/bin/env bash
# Project-local Express dev environment (venv + config under ./.config).
# Usage: source scripts/dev-env.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating venv and installing express[dev]…"
  if command -v uv >/dev/null 2>&1; then
    uv sync --extra dev
  else
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install -U pip
    pip install -e ".[dev]"
  fi
fi

# shellcheck disable=SC1091
source .venv/bin/activate

ENV_FILE="$ROOT/.env.local"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE — copy .env.local.example and fill in S3 credentials."
  return 1 2>/dev/null || exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

export XDG_CONFIG_HOME="$ROOT/.config"
mkdir -p "$XDG_CONFIG_HOME/express"

python <<PY
import os
from pathlib import Path

from express.base.config import save_config

root = Path("${ROOT}")
workdir = os.environ.get("LOCAL_WORKDIR") or str(root / ".local-models")
os.environ["LOCAL_WORKDIR"] = workdir

payload = {
    "S3_AK": os.environ["S3_AK"],
    "S3_SK": os.environ["S3_SK"],
    "S3_ENDPOINT": os.environ["S3_ENDPOINT"],
    "S3_BUCKET": os.environ["S3_BUCKET"],
    "LOCAL_WORKDIR": workdir,
}
path = save_config(payload)
Path(workdir).mkdir(parents=True, exist_ok=True)
print(f"Express config: {path}")
print(f"Local models:   {workdir}")
PY

export LOCAL_WORKDIR="${LOCAL_WORKDIR:-$ROOT/.local-models}"
echo "venv: $(which python)"
echo "CLI:  express --help"
echo "Tab:  express completion install   # once per shell"
