#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/cloud_llamafactory_readme.sh preprocess|micro|smoke|full|full-cached

Runs the README-aligned LLaMA-Factory DPO profile through the cloud byf conda
environment. The smoke profile only adds max_steps for timing; the micro
profile also caps max_samples to make a quick step-time probe. Both keep the
same model, dataset, LoRA/DPO settings, and pref_loss as the README profile.
The preprocess profile saves the tokenized dataset without loading the model.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/cloud/byf_env.sh"

MODE="${1:-smoke}"
case "$MODE" in
  preprocess)
    CONFIG="$REPO_ROOT/configs/llamafactory/train_dpo_qwen2_5_vl_readme_preprocess.yaml"
    LF_CONFIG="train_dpo_qwen2_5_vl_preprocess.yaml"
    ;;
  micro)
    CONFIG="$REPO_ROOT/configs/llamafactory/train_dpo_qwen2_5_vl_readme_micro.yaml"
    LF_CONFIG="train_dpo_qwen2_5_vl_micro.yaml"
    ;;
  smoke)
    CONFIG="$REPO_ROOT/configs/llamafactory/train_dpo_qwen2_5_vl_readme_smoke.yaml"
    LF_CONFIG="train_dpo_qwen2_5_vl_smoke.yaml"
    ;;
  full)
    CONFIG="$REPO_ROOT/configs/llamafactory/train_dpo_qwen2_5_vl_readme.yaml"
    LF_CONFIG="train_dpo_qwen2_5_vl.yaml"
    ;;
  full-cached)
    CONFIG="$REPO_ROOT/configs/llamafactory/train_dpo_qwen2_5_vl_readme_full_cached.yaml"
    LF_CONFIG="train_dpo_qwen2_5_vl_full_cached.yaml"
    ;;
  --help|-h)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ ! -x "$CLOUD_LLAMAFACTORY_CLI" ]]; then
  echo "Missing LLaMA-Factory CLI: $CLOUD_LLAMAFACTORY_CLI" >&2
  exit 1
fi

cd "$REPO_ROOT/LlamaFactory"
cp "$CONFIG" "$LF_CONFIG"

if [[ -n "${PREPROCESSING_NUM_WORKERS:-}" ]]; then
  "$CLOUD_PYTHON" - "$LF_CONFIG" "$PREPROCESSING_NUM_WORKERS" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
workers = sys.argv[2]
text = path.read_text()
text = re.sub(r"(?m)^preprocessing_num_workers:\s*\d+\s*$", f"preprocessing_num_workers: {workers}", text)
path.write_text(text)
PY
fi

if [[ "$MODE" == "preprocess" ]]; then
  exec "$CLOUD_PYTHON" "$REPO_ROOT/scripts/cloud_preprocess_llamafactory.py" "$LF_CONFIG"
fi

exec "$CLOUD_LLAMAFACTORY_CLI" train "$LF_CONFIG"
