# Local Trainable Project Setup

This checkout has been combined with the recovered 5,059-pair DPO annotation artifacts and local course images.

## Local data layout

The following paths are local-only and ignored by git:

- `data/raw` -> `../../data/raw`
- `data/processed` -> `../../data/processed`
- `data/audit` -> `../../data/audit`
- `images` -> `../项目文件/images`
- `llava/data/eval/AMBER/images` -> `../../../../../项目文件/llava/data/eval/AMBER/images`

The canonical DPO dataset inside this repository is:

```text
data/processed/qwen2_5_vl_dpo_full.json
```

It points to:

```text
data/processed/qwen2_5_vl_dpo_5059_54mini_54refine_recovered.json
```

Summary:

- DPO rows: 5,059
- Unique images: 5,059
- Raw answer rows: 100,000
- Training images available through `images/`: 10,000
- AMBER images available through `llava/data/eval/AMBER/images/`: 1,004

## LLaMA-Factory checkout

`LlamaFactory/` is a local ignored checkout of `hiyouga/LLaMA-Factory`.

Registered dataset:

```text
qwen2_5_vl_dpo_full
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

The Google Drive link provided for the complete annotations was not reachable from this environment. Direct access to `drive.google.com` timed out, and browser access returned unauthorized. The local recovered annotation artifacts already present in the workspace were used for this setup.

