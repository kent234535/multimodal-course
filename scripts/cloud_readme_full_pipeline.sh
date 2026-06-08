#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/cloud_readme_full_pipeline.sh

Runs the README-required full workflow on the cloud:
1. validate data, images, AMBER files, and selected free GPUs
2. train Qwen/Qwen2.5-VL-3B-Instruct with LLaMA-Factory DPO LoRA
3. apply the README LoRA adapter key fix
4. evaluate the base model and LoRA model on AMBER with existing scripts
5. write reports/cloud_runs/<run_id>/report.md and summary.json

Environment overrides:
  RUN_ID                 default: current timestamp
  GPU_LIST               default: 0,7
  TRAIN_MODE             default: full
  MASTER_PORT            default: 29617
  BASE_OUTPUT            default: amber_base_readme_<run_id>
  LORA_OUTPUT            default: amber_lora_readme_<run_id>
  SKIP_BASE_EVAL=1       skip base AMBER evaluation only
  SKIP_LORA_EVAL=1       skip LoRA AMBER evaluation only
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/cloud/byf_env.sh"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0,7}"
TRAIN_MODE="${TRAIN_MODE:-full}"
MASTER_PORT="${MASTER_PORT:-29617}"
BASE_OUTPUT="${BASE_OUTPUT:-amber_base_readme_${RUN_ID}}"
LORA_OUTPUT="${LORA_OUTPUT:-amber_lora_readme_${RUN_ID}}"

RUN_DIR="$REPO_ROOT/reports/cloud_runs/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

PIPELINE_LOG="$RUN_DIR/pipeline.log"
START_TIME="$(date -Is)"
PIPELINE_STATUS="running"
FAILURE_REASON=""

exec > >(tee -a "$PIPELINE_LOG") 2>&1

finish() {
  local exit_code=$?
  local end_time
  end_time="$(date -Is)"
  if [[ "$exit_code" -eq 0 && "$PIPELINE_STATUS" == "running" ]]; then
    PIPELINE_STATUS="completed"
  elif [[ "$exit_code" -ne 0 && "$PIPELINE_STATUS" == "running" ]]; then
    PIPELINE_STATUS="failed"
    FAILURE_REASON="${FAILURE_REASON:-pipeline exited with code $exit_code}"
  fi
  "$CLOUD_PYTHON" "$REPO_ROOT/scripts/cloud_write_full_report.py" \
    --repo-root "$REPO_ROOT" \
    --run-dir "$RUN_DIR" \
    --run-id "$RUN_ID" \
    --status "$PIPELINE_STATUS" \
    --exit-code "$exit_code" \
    --gpu-list "$GPU_LIST" \
    --start-time "$START_TIME" \
    --end-time "$end_time" \
    --train-config "$REPO_ROOT/configs/llamafactory/train_dpo_qwen2_5_vl_readme.yaml" \
    --train-mode "$TRAIN_MODE" \
    --base-output "$BASE_OUTPUT" \
    --lora-output "$LORA_OUTPUT" \
    --adapter-dir "LlamaFactory/saves/qwen2.5-vl-3b-lora-dpo" \
    --fixed-adapter-dir "LlamaFactory/saves/qwen2.5-vl-3b-lora-dpo_fixed" \
    --failure "$FAILURE_REASON" || true
  echo "pipeline_status=$PIPELINE_STATUS"
  echo "report=$RUN_DIR/report.md"
  exit "$exit_code"
}
trap finish EXIT

die() {
  FAILURE_REASON="$*"
  echo "ERROR: $FAILURE_REASON" >&2
  exit 1
}

run_step() {
  local name="$1"
  local log_path="$2"
  shift 2
  echo
  echo "=== START $name $(date -Is) ==="
  echo "log=$log_path"
  set +e
  "$@" > >(tee -a "$log_path") 2>&1
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    PIPELINE_STATUS="failed"
    FAILURE_REASON="$name failed with exit code $rc"
    echo "=== FAIL $name rc=$rc $(date -Is) ==="
    return "$rc"
  fi
  echo "=== DONE $name $(date -Is) ==="
}

gpu_count() {
  local list="$1"
  awk -F',' '{print NF}' <<<"$list"
}

validate_selected_gpus() {
  "$CLOUD_PYTHON" - "$GPU_LIST" <<'PY'
import subprocess
import sys

gpu_list = [int(x) for x in sys.argv[1].split(",") if x.strip()]
query = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
stats = {}
for line in query.strip().splitlines():
    idx, mem, util = [part.strip() for part in line.split(",")]
    stats[int(idx)] = {"memory_used_mb": int(mem), "utilization_gpu": int(util)}

bad = []
for idx in gpu_list:
    item = stats.get(idx)
    if item is None:
        bad.append(f"GPU{idx}: missing")
    elif item["memory_used_mb"] > 1024:
        bad.append(f"GPU{idx}: {item['memory_used_mb']} MiB used")

print("selected_gpu_stats=", {idx: stats.get(idx) for idx in gpu_list})
if bad:
    raise SystemExit("selected GPUs are not free enough: " + "; ".join(bad))
PY
}

validate_inputs() {
  "$CLOUD_PYTHON" - "$REPO_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dataset_info = json.loads((root / "LlamaFactory/data/dataset_info.json").read_text())
assert "qwen2.5_vl_3b" in dataset_info, "missing qwen2.5_vl_3b in dataset_info.json"
dataset_path = root / "LlamaFactory/data/qwen2.5-vl-3b.json"
rows = json.loads(dataset_path.read_text())
assert len(rows) == 10000, f"expected 10000 DPO rows, got {len(rows)}"
assert all(row.get("images") for row in rows[:10]), "first rows are missing images"
image_dir = root / "LlamaFactory/images"
image_count = len(list(image_dir.glob("*.jpg")))
assert image_count == 10000, f"expected 10000 training images, got {image_count}"

amber = root / "llava/data/eval/AMBER"
for rel in [
    "data/query/query_generative.json",
    "data/query/query_discriminative.json",
    "data/annotations.json",
    "data/relation.json",
    "data/metrics.txt",
    "data/safe_words.txt",
]:
    assert (amber / rel).exists(), f"missing AMBER file: {rel}"
gen = json.loads((amber / "data/query/query_generative.json").read_text())
dis = json.loads((amber / "data/query/query_discriminative.json").read_text())
amber_images = len(list((amber / "images").glob("*.jpg")))
assert len(gen) == 1004, f"expected 1004 generative queries, got {len(gen)}"
assert len(dis) == 14216, f"expected 14216 discriminative queries, got {len(dis)}"
assert amber_images == 1004, f"expected 1004 AMBER images, got {amber_images}"
print({"dpo_rows": len(rows), "training_images": image_count, "amber_gen": len(gen), "amber_dis": len(dis), "amber_images": amber_images})
PY
}

echo "run_id=$RUN_ID"
echo "repo_root=$REPO_ROOT"
echo "run_dir=$RUN_DIR"
echo "gpu_list=$GPU_LIST"
echo "train_mode=$TRAIN_MODE"
echo "start_time=$START_TIME"
echo "python=$CLOUD_PYTHON"
echo "llamafactory_cli=$CLOUD_LLAMAFACTORY_CLI"

validate_selected_gpus
validate_inputs

case "$TRAIN_MODE" in
  full)
    ;;
  full-cached)
    ;;
  *)
    die "TRAIN_MODE must be full or full-cached for this pipeline"
    ;;
esac

NPROC_PER_NODE="$(gpu_count "$GPU_LIST")"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE
export MASTER_PORT
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "MASTER_PORT=$MASTER_PORT"

run_step "train_${TRAIN_MODE}" "$LOG_DIR/train.log" \
  bash "$REPO_ROOT/scripts/cloud_llamafactory_readme.sh" "$TRAIN_MODE"

run_step "fix_lora_adapter" "$LOG_DIR/fix_lora.log" \
  "$CLOUD_PYTHON" "$REPO_ROOT/scripts/fix_lora_adapter.py" \
    --adapter-dir "$REPO_ROOT/LlamaFactory/saves/qwen2.5-vl-3b-lora-dpo" \
    --output-dir "$REPO_ROOT/LlamaFactory/saves/qwen2.5-vl-3b-lora-dpo_fixed" \
    --overwrite

export GPU_LIST
if [[ "${SKIP_BASE_EVAL:-0}" != "1" ]]; then
  run_step "eval_amber_base" "$LOG_DIR/eval_base.log" \
    bash "$REPO_ROOT/scripts/eval_amber.sh" \
      --profile full \
      --mode base \
      --output-name "$BASE_OUTPUT" \
      --model-name "Qwen/Qwen2.5-VL-3B-Instruct"
fi

if [[ "${SKIP_LORA_EVAL:-0}" != "1" ]]; then
  run_step "eval_amber_lora" "$LOG_DIR/eval_lora.log" \
    bash "$REPO_ROOT/scripts/eval_amber.sh" \
      --profile full \
      --mode lora \
      --output-name "$LORA_OUTPUT" \
      --model-name "Qwen/Qwen2.5-VL-3B-Instruct" \
      --lora-name "LlamaFactory/saves/qwen2.5-vl-3b-lora-dpo_fixed"
fi

"$CLOUD_PYTHON" "$REPO_ROOT/scripts/summarize_amber_results.py" \
  --base-output "$BASE_OUTPUT" \
  --lora-output "$LORA_OUTPUT" \
  > "$RUN_DIR/amber_summary.md" || true
"$CLOUD_PYTHON" "$REPO_ROOT/scripts/summarize_amber_results.py" \
  --base-output "$BASE_OUTPUT" \
  --lora-output "$LORA_OUTPUT" \
  --json > "$RUN_DIR/amber_summary.json" || true

echo "pipeline completed; report will be written by EXIT trap"
