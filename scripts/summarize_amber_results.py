#!/usr/bin/env python3
"""Summarize AMBER base-vs-LoRA metrics from existing evaluation JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GEN_METRICS = ("CHAIR", "Cover", "Hal", "Cog")
DIS_METRICS = ("Accuracy", "F1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract AMBER generative and discriminative metrics for base vs LoRA outputs.")
    parser.add_argument("--base-output", default="amber_base_local", help="Base output name used by AMBER scripts, without suffix.")
    parser.add_argument("--lora-output", default="amber_lora_local", help="LoRA output name used by AMBER scripts, without suffix.")
    parser.add_argument("--amber-dir", default="llava/data/eval/AMBER", help="AMBER directory.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a Markdown table.")
    return parser.parse_args()


def read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
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


def metrics_for_output(amber_dir: Path, output_name: str) -> dict[str, Any]:
    gen_path = amber_dir / "amber_gen" / "answers" / f"{output_name}_eval_amber.jsonl"
    dis_path = amber_dir / "amber_dis" / "answers" / f"{output_name}_eval_amber.jsonl"
    gen_section = find_section(read_jsonl_objects(gen_path), "Generative Task")
    dis_section = find_section(read_jsonl_objects(dis_path), "Descriminative Task")
    return {
        "files": {"generative": str(gen_path), "discriminative": str(dis_path)},
        "generative": {metric: gen_section.get(metric) for metric in GEN_METRICS},
        "discriminative": {metric: dis_section.get(metric) for metric in DIS_METRICS},
    }


def format_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def print_markdown(summary: dict[str, Any]) -> None:
    print("# AMBER Result Summary")
    print()
    print("| Split | Metric | Base | LoRA |")
    print("|---|---:|---:|---:|")
    for section, metrics in (("Generative", GEN_METRICS), ("Discriminative", DIS_METRICS)):
        key = section.lower()
        for metric in metrics:
            base = summary["base"][key].get(metric)
            lora = summary["lora"][key].get(metric)
            print(f"| {section} | {metric} | {format_value(base)} | {format_value(lora)} |")
    print()
    print("Files checked:")
    for label in ("base", "lora"):
        print(f"- {label} generative: `{summary[label]['files']['generative']}`")
        print(f"- {label} discriminative: `{summary[label]['files']['discriminative']}`")


def main() -> int:
    args = parse_args()
    amber_dir = Path(args.amber_dir)
    summary = {
        "base": metrics_for_output(amber_dir, args.base_output),
        "lora": metrics_for_output(amber_dir, args.lora_output),
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_markdown(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
