#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/dev-env.sh"
MODEL="$ROOT/examples/tiny-safetensors-model"
mkdir -p "$MODEL"

python <<PY
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file

out = Path("$MODEL") / "model.safetensors"
save_file(
    {
        "layer.weight": np.random.randn(8, 8).astype(np.float32) * 0.02,
        "layer.bias": np.zeros(8, dtype=np.float32),
    },
    out,
)
print("Wrote", out)
PY

if [[ ! -f "$MODEL/metadata.json" ]]; then
  python <<PY
from pathlib import Path
from express.base.model import Model

m = Model(Path("$MODEL"))
m.create_metadata(
    name="tiny-safetensors-model",
    version="0.0.1",
    authors="local-dev",
    tags=["dev", "safetensors"],
)
m.write_index_file()
print("Metadata + index ready under $MODEL")
PY
fi

echo "Try: express push $MODEL"
