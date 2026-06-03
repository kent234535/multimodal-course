# LLaMA-Factory integration notes

## Dataset registration

After generating a dataset, register and optionally copy it into `${LLAMAFACTORY_DIR}/data/` with:

```bash
pixi run register-dpo-local
```

This calls `scripts/register_llamafactory_dataset.py` and merges the dataset entry into `${LLAMAFACTORY_DIR}/data/dataset_info.json` so manual JSON edits are not required for the normal local workflow. The pixi task also creates `${LLAMAFACTORY_DIR}/images` as a symlink to this repository's `images/` directory when needed, so dataset rows like `"images": ["images/1.jpg"]` resolve under LLaMA-Factory.

For review, the committed registration snippet is:

```bash
configs/llamafactory/dataset_info.qwen2_5_vl_dpo_local_1000.json
```

The expected LLaMA-Factory dataset row is `qwen2_5_vl_dpo_local_1000` and maps:

- `instruction` -> prompt
- `input` -> query
- `chosen` -> chosen
- `rejected` -> rejected
- `images` -> images

## Training profiles

- Local RTX 4070S profile: `configs/llamafactory/train_dpo_qwen2_5_vl_local.yaml`
- Full 8xA100 profile: `configs/llamafactory/train_dpo_qwen2_5_vl_full.yaml`

Both profiles preserve the assignment constraints:

- `stage: dpo`
- `finetuning_type: lora`
- `pref_loss: sigmoid`
- `template: qwen2_vl`
- `model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct`
