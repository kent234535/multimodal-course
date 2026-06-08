#!/usr/bin/env python3
"""Write a cloud README-aligned training/evaluation report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_CONFIG = {
    "model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
    "stage": "dpo",
    "do_train": "true",
    "finetuning_type": "lora",
    "pref_loss": "sigmoid",
    "dataset": "qwen2.5_vl_3b",
    "template": "qwen2_vl",
    "cutoff_len": "2048",
    "max_samples": "100000",
    "per_device_train_batch_size": "1",
    "gradient_accumulation_steps": "16",
    "learning_rate": "4.0e-6",
    "num_train_epochs": "2.0",
    "bf16": "true",
    "flash_attn": "fa2",
}

GEN_METRICS = ("CHAIR", "Cover", "Hal", "Cog")
DIS_METRICS = ("Accuracy", "Precision", "Recall", "F1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--exit-code", required=True)
    parser.add_argument("--gpu-list", required=True)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--end-time", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--train-mode", required=True)
    parser.add_argument("--base-output", required=True)
    parser.add_argument("--lora-output", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--fixed-adapter-dir", required=True)
    parser.add_argument("--failure", default="")
    return parser.parse_args()


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if limit is None else text[-limit:]


def read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in read_text(path).splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def count_json_array(path: Path) -> int | None:
    payload = read_json(path)
    if isinstance(payload, list):
        return len(payload)
    return None


def count_jsonl(path: Path) -> int | None:
    if not path.is_file():
        return None
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


def find_section(records: list[dict[str, Any]], section_name: str) -> dict[str, Any]:
    for record in reversed(records):
        section = record.get(section_name)
        if isinstance(section, dict):
            return section
    return {}


def amber_metrics(repo_root: Path, output_name: str) -> dict[str, Any]:
    amber_dir = repo_root / "llava/data/eval/AMBER"
    gen_answer = amber_dir / "amber_gen/answers" / f"{output_name}.jsonl"
    dis_answer = amber_dir / "amber_dis/answers" / f"{output_name}.jsonl"
    gen_eval = amber_dir / "amber_gen/answers" / f"{output_name}_eval_amber.jsonl"
    dis_eval = amber_dir / "amber_dis/answers" / f"{output_name}_eval_amber.jsonl"
    return {
        "answers": {
            "generative": str(gen_answer),
            "discriminative": str(dis_answer),
            "generative_count": count_jsonl(gen_answer),
            "discriminative_count": count_jsonl(dis_answer),
        },
        "eval_files": {"generative": str(gen_eval), "discriminative": str(dis_eval)},
        "generative": find_section(read_jsonl_objects(gen_eval), "Generative Task"),
        "discriminative": find_section(read_jsonl_objects(dis_eval), "Descriminative Task"),
    }


def last_trainer_metrics(adapter_dir: Path) -> dict[str, Any]:
    trainer_state = read_json(adapter_dir / "trainer_state.json")
    if not isinstance(trainer_state, dict):
        return {}

    log_history = trainer_state.get("log_history")
    train_rows = [row for row in log_history or [] if isinstance(row, dict)]
    result_row = {}
    for row in reversed(train_rows):
        if any(key.startswith("train_") for key in row):
            result_row = row
            break
    return {
        "global_step": trainer_state.get("global_step"),
        "best_metric": trainer_state.get("best_metric"),
        "last_train_row": result_row,
    }


def iso_duration(start: str, end: str) -> str:
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError:
        return "unknown"
    seconds = int((end_dt - start_dt).total_seconds())
    if seconds < 0:
        return "unknown"
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours}h {minutes}m {sec}s"


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def metric_delta(metric: str, base: Any, lora: Any) -> str:
    if not isinstance(base, (int, float)) or not isinstance(lora, (int, float)):
        return "missing"
    delta = lora - base
    if metric in {"CHAIR", "Hal", "Cog"}:
        direction = "better" if delta < 0 else "worse" if delta > 0 else "same"
    else:
        direction = "better" if delta > 0 else "worse" if delta < 0 else "same"
    return f"{delta:+.4g} ({direction})"


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_config = Path(args.train_config)
    config_values = parse_simple_yaml(train_config)
    config_check = {
        key: {
            "expected": expected,
            "actual": config_values.get(key),
            "match": str(config_values.get(key)).lower() == expected.lower(),
        }
        for key, expected in EXPECTED_CONFIG.items()
    }

    data_file = repo_root / "LlamaFactory/data/qwen2.5-vl-3b.json"
    amber_query_dir = repo_root / "llava/data/eval/AMBER/data/query"
    adapter_dir = repo_root / args.adapter_dir
    fixed_adapter_dir = repo_root / args.fixed_adapter_dir
    base_metrics = amber_metrics(repo_root, args.base_output)
    lora_metrics = amber_metrics(repo_root, args.lora_output)

    summary = {
        "run_id": args.run_id,
        "status": args.status,
        "exit_code": args.exit_code,
        "failure": args.failure,
        "gpu_list": args.gpu_list,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "duration": iso_duration(args.start_time, args.end_time),
        "train_mode": args.train_mode,
        "train_config": str(train_config),
        "readme_alignment": config_check,
        "dataset_rows": count_json_array(data_file),
        "amber_queries": {
            "generative": count_json_array(amber_query_dir / "query_generative.json"),
            "discriminative": count_json_array(amber_query_dir / "query_discriminative.json"),
        },
        "adapter": {
            "dir": str(adapter_dir),
            "exists": adapter_dir.is_dir(),
            "fixed_dir": str(fixed_adapter_dir),
            "fixed_exists": fixed_adapter_dir.is_dir(),
            "trainer": last_trainer_metrics(adapter_dir),
        },
        "amber": {"base": base_metrics, "lora": lora_metrics},
        "logs": {
            "pipeline": str(run_dir / "pipeline.log"),
            "train": str(run_dir / "logs/train.log"),
            "fix_lora": str(run_dir / "logs/fix_lora.log"),
            "eval_base": str(run_dir / "logs/eval_base.log"),
            "eval_lora": str(run_dir / "logs/eval_lora.log"),
            "report": str(run_dir / "report.md"),
        },
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# Cloud README Full Training And AMBER Report")
    lines.append("")
    lines.append(f"- Run ID: `{args.run_id}`")
    lines.append(f"- Status: `{args.status}`")
    lines.append(f"- Exit code: `{args.exit_code}`")
    lines.append(f"- GPU_LIST: `{args.gpu_list}`")
    lines.append(f"- Duration: `{summary['duration']}`")
    lines.append(f"- Train mode: `{args.train_mode}`")
    if args.failure:
        lines.append(f"- Failure: `{args.failure}`")
    lines.append("")
    lines.append("## README Alignment")
    lines.append("")
    lines.append("| Key | Expected | Actual | Match |")
    lines.append("|---|---|---|---:|")
    for key, item in config_check.items():
        lines.append(f"| `{key}` | `{item['expected']}` | `{fmt(item['actual'])}` | {item['match']} |")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- Training DPO rows: `{fmt(summary['dataset_rows'])}`")
    lines.append(f"- AMBER generative queries: `{fmt(summary['amber_queries']['generative'])}`")
    lines.append(f"- AMBER discriminative queries: `{fmt(summary['amber_queries']['discriminative'])}`")
    lines.append("")
    lines.append("## Training")
    lines.append("")
    lines.append(f"- Adapter dir: `{summary['adapter']['dir']}`")
    lines.append(f"- Fixed adapter dir: `{summary['adapter']['fixed_dir']}`")
    trainer = summary["adapter"]["trainer"]
    if trainer:
        lines.append(f"- Global step: `{fmt(trainer.get('global_step'))}`")
        last_row = trainer.get("last_train_row") or {}
        for key in ("train_runtime", "train_samples_per_second", "train_steps_per_second", "train_loss", "epoch"):
            if key in last_row:
                lines.append(f"- {key}: `{fmt(last_row[key])}`")
    else:
        lines.append("- Trainer state: `missing`")
    lines.append("")
    lines.append("## AMBER Metrics")
    lines.append("")
    lines.append("| Split | Metric | Base | LoRA | Delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for metric in GEN_METRICS:
        base = base_metrics["generative"].get(metric)
        lora = lora_metrics["generative"].get(metric)
        lines.append(f"| Generative | {metric} | {fmt(base)} | {fmt(lora)} | {metric_delta(metric, base, lora)} |")
    for metric in DIS_METRICS:
        base = base_metrics["discriminative"].get(metric)
        lora = lora_metrics["discriminative"].get(metric)
        lines.append(f"| Discriminative | {metric} | {fmt(base)} | {fmt(lora)} | {metric_delta(metric, base, lora)} |")
    lines.append("")
    lines.append("## Answer Counts")
    lines.append("")
    lines.append("| Output | Generative | Discriminative |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| Base | {fmt(base_metrics['answers']['generative_count'])} | {fmt(base_metrics['answers']['discriminative_count'])} |"
    )
    lines.append(
        f"| LoRA | {fmt(lora_metrics['answers']['generative_count'])} | {fmt(lora_metrics['answers']['discriminative_count'])} |"
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for label, path in summary["logs"].items():
        lines.append(f"- {label}: `{path}`")
    lines.append(f"- summary_json: `{run_dir / 'summary.json'}`")
    lines.append("")
    lines.append("## Tail")
    lines.append("")
    for label in ("train", "eval_base", "eval_lora"):
        path = Path(summary["logs"][label])
        tail = read_text(path, limit=4000)
        if not tail:
            continue
        lines.append(f"### {label}")
        lines.append("")
        lines.append("```text")
        lines.append(tail.rstrip())
        lines.append("```")
        lines.append("")

    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report: {run_dir / 'report.md'}")
    print(f"Wrote summary: {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
