#!/usr/bin/env python3
"""Register a generated DPO dataset in LLaMA-Factory's dataset_info.json."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

DEFAULT_COLUMNS = {
    "prompt": "instruction",
    "query": "input",
    "chosen": "chosen",
    "rejected": "rejected",
    "images": "images",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a DPO dataset entry into LLaMA-Factory data/dataset_info.json.")
    parser.add_argument("--llamafactory-dir", default="LlamaFactory", help="Path to the LLaMA-Factory checkout.")
    parser.add_argument("--dataset-name", default="qwen2_5_vl_dpo_local_1000", help="LLaMA-Factory dataset registry key.")
    parser.add_argument("--dataset-file", default="qwen2_5_vl_dpo_local_1000.json", help="Dataset JSON file name under LLaMA-Factory/data.")
    parser.add_argument("--source-json", default=None, help="Optional generated dataset to copy into LLaMA-Factory/data/.")
    parser.add_argument("--image-source", default="images", help="Repository image directory to link into LLaMA-Factory when --link-images is set.")
    parser.add_argument("--link-images", action="store_true", help="Create LLaMA-Factory/images as a symlink to --image-source if it does not already exist.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing dataset file/registry entry.")
    return parser.parse_args()


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json_object(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def main() -> int:
    args = parse_args()
    llama_dir = Path(args.llamafactory_dir)
    data_dir = llama_dir / "data"
    dataset_info_path = data_dir / "dataset_info.json"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"LLaMA-Factory data directory not found: {data_dir}")
    if not dataset_info_path.is_file():
        raise FileNotFoundError(f"LLaMA-Factory dataset_info.json not found: {dataset_info_path}")

    target_dataset_path = data_dir / args.dataset_file
    if args.source_json:
        source_path = Path(args.source_json)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source dataset not found: {source_path}")
        if target_dataset_path.exists() and not args.overwrite:
            raise FileExistsError(f"Dataset already exists: {target_dataset_path}. Use --overwrite to replace it.")
        shutil.copy2(source_path, target_dataset_path)

    if args.link_images:
        image_source = Path(args.image_source).resolve()
        image_target = llama_dir / "images"
        if not image_source.is_dir():
            raise FileNotFoundError(f"Image source directory not found: {image_source}")
        if image_target.exists():
            if not image_target.is_dir():
                raise FileExistsError(f"LLaMA-Factory image target exists but is not a directory: {image_target}")
        else:
            image_target.symlink_to(image_source, target_is_directory=True)
            print(f"Linked {image_target} -> {image_source}")

    dataset_info = read_json_object(dataset_info_path)
    if args.dataset_name in dataset_info and not args.overwrite:
        raise KeyError(f"Dataset entry already exists: {args.dataset_name}. Use --overwrite to replace it.")
    dataset_info[args.dataset_name] = {
        "file_name": args.dataset_file,
        "columns": DEFAULT_COLUMNS,
        "ranking": True,
    }
    write_json_object(dataset_info_path, dataset_info)
    print(f"Registered {args.dataset_name} in {dataset_info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
