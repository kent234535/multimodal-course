# DPO sample previews

Small preview files in this directory are safe to commit and are intended only for schema review/smoke tests.

Generate a no-API mock preview with:

```bash
pixi run build-dpo-preview
```

or directly:

```bash
python3 scripts/build_dpo_pairs.py \
  --answers data/raw/answers.jsonl \
  --output data/samples/qwen2_5_vl_dpo_preview.json \
  --audit data/samples/qwen2_5_vl_dpo_preview_audit.jsonl \
  --judge-mode mock \
  --max-accepted 5 \
  --max-questions 5 \
  --overwrite
```

Full generated datasets and audit files should go under `data/processed/` and `data/audit/`, which are ignored by git.
