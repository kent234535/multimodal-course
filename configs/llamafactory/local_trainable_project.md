# Local Trainable Project Setup

This checkout has been combined with the complete 10,000-pair DPO annotation artifacts from the provided Google Drive package and local course images.

## Local data layout

The following paths are local-only and ignored by git:

- `data/raw` -> `../../data/raw`
- `data/processed` -> `../../data/processed`
- `data/audit` -> `../../data/audit`
- `images` -> `../项目文件/images`
- `llava/data/eval/AMBER/images` -> `../../../../../项目文件/llava/data/eval/AMBER/images`

The canonical DPO dataset inside this repository is:

```text
data/processed/qwen2_5_vl_dpo.json
```

Compatibility alias:

```text
data/processed/qwen2_5_vl_dpo_full.json -> qwen2_5_vl_dpo.json
```

Summary:

- DPO rows: 10,000
- Unique images: 10,000
- Raw answer rows: 100,000
- Training images available through `images/`: 10,000
- AMBER images available through `llava/data/eval/AMBER/images/`: 1,004
- Judge model: `gpt-5.4`
- Refine model: `gpt-5.5`
- Average confidence: 0.911256

The earlier recovered 5,059-pair artifacts are retained locally under their original `qwen2_5_vl_dpo_5059_54mini_54refine_recovered*` filenames for audit/history, but they are no longer the default full-training dataset.

## LLaMA-Factory checkout

`LlamaFactory/` is a local ignored checkout of `hiyouga/LLaMA-Factory`.

Registered dataset:

```text
qwen2_5_vl_dpo_full
```

Canonical registered dataset:

```text
qwen2_5_vl_dpo
```

Registered file:

```text
LlamaFactory/data/qwen2_5_vl_dpo_full.json
```

Image link:

```text
LlamaFactory/images -> /Users/kent/Desktop/课程/多模态学习导论/项目文件/images
```

## Training command

After installing LLaMA-Factory dependencies, run from this repository:

```bash
REPO_ROOT=$(pwd)
cd LlamaFactory
llamafactory-cli train "$REPO_ROOT/configs/llamafactory/train_dpo_qwen2_5_vl_full.yaml"
```

The full training profile uses:

- `model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct`
- `stage: dpo`
- `finetuning_type: lora`
- `pref_loss: sigmoid`
- `dataset: qwen2_5_vl_dpo_full`

## Notes

The Google Drive package was downloaded locally through the system proxy on 2026-06-08. The downloaded package SHA-256 is:

```text
b3a85c90e0f272e840f0b6607a9098a83d2260e2b1fdccc3db49eccc4f0d4d1d
```
