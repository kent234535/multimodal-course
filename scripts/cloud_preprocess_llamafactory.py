#!/usr/bin/env python3
"""Save a LLaMA-Factory tokenized dataset without loading the model."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: cloud_preprocess_llamafactory.py CONFIG.yaml", file=sys.stderr)
        return 2

    lf_root = Path.cwd()
    sys.path.insert(0, str(lf_root / "src"))

    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams.parser import _parse_train_args, _set_env_vars, _set_transformers_logging
    from llamafactory.model import load_tokenizer

    config_path = Path(sys.argv[1])
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    model_args, data_args, training_args, finetuning_args, _ = _parse_train_args(config)

    if not data_args.tokenized_path:
        raise ValueError("tokenized_path must be set for preprocessing.")

    tokenized_path = Path(data_args.tokenized_path)
    if os.environ.get("PREPROCESS_CLEAN", "0") == "1" and tokenized_path.exists():
        shutil.rmtree(tokenized_path)

    _set_transformers_logging()
    _set_env_vars()

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    stage = "rm" if finetuning_args.stage == "dpo" else finetuning_args.stage
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage=stage, **tokenizer_module)

    train_dataset = dataset_module.get("train_dataset")
    train_size = len(train_dataset) if train_dataset is not None and hasattr(train_dataset, "__len__") else "unknown"
    print(f"tokenized_path={data_args.tokenized_path}")
    print(f"train_dataset_size={train_size}")
    print("preprocess_complete=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
