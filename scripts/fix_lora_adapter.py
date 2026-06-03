#!/usr/bin/env python3
"""Fix LoRA adapter tensor keys for the AMBER evaluation loader.

This implements the README's key replacement logic:
`.language_model.` -> `.` inside `adapter_model.safetensors`, while copying the
rest of the adapter directory unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy and fix a Qwen2.5-VL LoRA adapter for evaluation.")
    parser.add_argument("--adapter-dir", required=True, help="Input LoRA adapter directory containing adapter_model.safetensors.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <adapter-dir>_fixed.")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing output directory before copying.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(str(adapter_dir) + "_fixed")
    tensor_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    if not tensor_path.is_file():
        raise FileNotFoundError(f"Adapter tensor file not found: {tensor_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)

    shutil.copytree(adapter_dir, output_dir, dirs_exist_ok=True)

    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required. Install the pixi environment or run `pip install safetensors`.") from exc

    tensors = load_file(str(tensor_path))
    fixed_tensors = {key.replace(".language_model.", "."): value for key, value in tensors.items()}
    save_file(fixed_tensors, str(output_dir / "adapter_model.safetensors"))

    with (output_dir / "adapter_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote fixed adapter to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
