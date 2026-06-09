# AMBER Result Summary

| Split | Metric | Base | LoRA |
|---|---:|---:|---:|
| Generative | CHAIR | 8.1 | 5.7 |
| Generative | Cover | 69.3 | 62.9 |
| Generative | Hal | 49.4 | 29.2 |
| Generative | Cog | 5.2 | 2.3 |
| Discriminative | Accuracy | 55.2 | 69.6 |
| Discriminative | F1 | 57.4 | 74.8 |

Files checked:
- base generative: `llava/data/eval/AMBER/amber_gen/answers/amber_base_readme_readme_full_20260609_004625_eval_amber.jsonl`
- base discriminative: `llava/data/eval/AMBER/amber_dis/answers/amber_base_readme_readme_full_20260609_004625_eval_amber.jsonl`
- lora generative: `llava/data/eval/AMBER/amber_gen/answers/amber_lora_readme_readme_full_20260609_004625_eval_amber.jsonl`
- lora discriminative: `llava/data/eval/AMBER/amber_dis/answers/amber_lora_readme_readme_full_20260609_004625_eval_amber.jsonl`
