#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/eval_amber.sh --profile local|full --mode base|lora --output-name NAME [--model-name MODEL] [--lora-name PATH]

Wraps the existing AMBER evaluation scripts without changing their evaluation
arguments. The wrapper only sets MODEL_NAME, LORA_NAME, OUTPUT_NAME, and GPU_LIST.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PROFILE="${AMBER_PROFILE:-local}"
MODE="${AMBER_MODE:-base}"
OUTPUT_NAME="${OUTPUT_NAME:-}"
MODEL_NAME_ARG="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"
LORA_NAME_ARG="${LORA_NAME:-${LORA_ADAPTER_DIR:-}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --output-name)
      OUTPUT_NAME="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME_ARG="$2"
      shift 2
      ;;
    --lora-name)
      LORA_NAME_ARG="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$PROFILE" == "local" ]]; then
  GPU_LIST_ARG="${GPU_LIST:-0}"
elif [[ "$PROFILE" == "full" ]]; then
  GPU_LIST_ARG="${GPU_LIST:-0,1,2,3,4,5,6,7}"
else
  echo "--profile must be local or full" >&2
  exit 2
fi

if [[ "$MODE" == "base" ]]; then
  LORA_NAME_ARG=""
  OUTPUT_NAME="${OUTPUT_NAME:-amber_base_${PROFILE}}"
elif [[ "$MODE" == "lora" ]]; then
  if [[ -z "$LORA_NAME_ARG" ]]; then
    LORA_NAME_ARG="LlamaFactory/saves/qwen2_5_vl_3b_lora_dpo_${PROFILE}_fixed"
  fi
  OUTPUT_NAME="${OUTPUT_NAME:-amber_lora_${PROFILE}}"
else
  echo "--mode must be base or lora" >&2
  exit 2
fi

export GPU_LIST="$GPU_LIST_ARG"
export MODEL_NAME="$MODEL_NAME_ARG"
export LORA_NAME="$LORA_NAME_ARG"
export OUTPUT_NAME

mkdir -p llava/data/eval/AMBER/amber_gen/answers llava/data/eval/AMBER/amber_dis/answers

echo "GPU_LIST: $GPU_LIST"
echo "MODEL_NAME: $MODEL_NAME"
echo "LORA_NAME: $LORA_NAME"
echo "OUTPUT_NAME: $OUTPUT_NAME"

bash llava/eval_script/eval_amber_gen.sh
bash llava/eval_script/eval_amber_dis.sh
