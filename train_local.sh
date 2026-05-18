#!/usr/bin/env bash

# Local/server training launcher without SLURM.
# Usage:
#   ./train_local.sh
#   ./train_local.sh configs/pred_len20.yaml
#   CONFIG=configs/default.yaml DEVICE=cuda ./train_local.sh

set -euo pipefail

CONFIG="${1:-${CONFIG:-configs/default.yaml}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_DISABLED="${WANDB_DISABLED:-true}"

export WANDB_MODE
export WANDB_DISABLED

if [ ! -f "$CONFIG" ]; then
  echo "[error] config not found: $CONFIG" >&2
  exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] python executable not found: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN=/path/to/python or activate your environment first." >&2
  exit 2
fi

echo "[local] config=$CONFIG"
echo "[local] python=$(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" --version

"$PYTHON_BIN" - <<'PY'
import torch

print(f"[local] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"[local] gpu={torch.cuda.get_device_name(0)} capability=sm_{cap[0]}{cap[1]}")
PY

if [ -n "$DEVICE" ]; then
  echo "[local] overriding training.device -> $DEVICE"
  "$PYTHON_BIN" scripts/train.py --config "$CONFIG" --device "$DEVICE"
else
  "$PYTHON_BIN" scripts/train.py --config "$CONFIG"
fi
