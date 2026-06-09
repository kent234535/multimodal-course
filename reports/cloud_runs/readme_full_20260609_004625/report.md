# Cloud README Full Training And AMBER Report

- Run ID: `readme_full_20260609_004625`
- Status: `completed_after_eval_resume`
- Exit code: `0`
- GPU_LIST: `0,7`
- Duration: `5h 23m 54s`
- Train mode: `full`

## README Alignment

| Key | Expected | Actual | Match |
|---|---|---|---:|
| `model_name_or_path` | `Qwen/Qwen2.5-VL-3B-Instruct` | `Qwen/Qwen2.5-VL-3B-Instruct` | True |
| `stage` | `dpo` | `dpo` | True |
| `do_train` | `true` | `true` | True |
| `finetuning_type` | `lora` | `lora` | True |
| `pref_loss` | `sigmoid` | `sigmoid` | True |
| `dataset` | `qwen2.5_vl_3b` | `qwen2.5_vl_3b` | True |
| `template` | `qwen2_vl` | `qwen2_vl` | True |
| `cutoff_len` | `2048` | `2048` | True |
| `max_samples` | `100000` | `100000` | True |
| `per_device_train_batch_size` | `1` | `1` | True |
| `gradient_accumulation_steps` | `16` | `16` | True |
| `learning_rate` | `4.0e-6` | `4.0e-6` | True |
| `num_train_epochs` | `2.0` | `2.0` | True |
| `bf16` | `true` | `true` | True |
| `flash_attn` | `fa2` | `fa2` | True |

## Data

- Training DPO rows: `10000`
- AMBER generative queries: `1004`
- AMBER discriminative queries: `14216`

## Training

- Adapter dir: `/home/byf/byf/multimodal/LlamaFactory/saves/qwen2.5-vl-3b-lora-dpo`
- Fixed adapter dir: `/home/byf/byf/multimodal/LlamaFactory/saves/qwen2.5-vl-3b-lora-dpo_fixed`
- Global step: `626`
- train_runtime: `1.004e+04`
- train_samples_per_second: `1.993`
- train_steps_per_second: `0.062`
- train_loss: `0.6551`
- epoch: `2`

## AMBER Metrics

| Split | Metric | Base | LoRA | Delta |
|---|---:|---:|---:|---:|
| Generative | CHAIR | 8.1 | 5.7 | -2.4 (better) |
| Generative | Cover | 69.3 | 62.9 | -6.4 (worse) |
| Generative | Hal | 49.4 | 29.2 | -20.2 (better) |
| Generative | Cog | 5.2 | 2.3 | -2.9 (better) |
| Discriminative | Accuracy | 55.2 | 69.6 | +14.4 (better) |
| Discriminative | Precision | 92.5 | 92.8 | +0.3 (better) |
| Discriminative | Recall | 41.6 | 62.6 | +21 (better) |
| Discriminative | F1 | 57.4 | 74.8 | +17.4 (better) |

## Answer Counts

| Output | Generative | Discriminative |
|---|---:|---:|
| Base | 1004 | 14216 |
| LoRA | 1004 | 14216 |

## Files

- pipeline: `reports/cloud_runs/readme_full_20260609_004625/pipeline.log`
- train: `reports/cloud_runs/readme_full_20260609_004625/logs/train.log`
- fix_lora: `reports/cloud_runs/readme_full_20260609_004625/logs/fix_lora.log`
- eval_base: `reports/cloud_runs/readme_full_20260609_004625/logs/eval_base.log`
- eval_lora: `reports/cloud_runs/readme_full_20260609_004625/logs/eval_lora.log`
- report: `reports/cloud_runs/readme_full_20260609_004625/report.md`
- summary_json: `reports/cloud_runs/readme_full_20260609_004625/summary.json`

## Tail

### train

```text
.5-vl-3b-lora-dpo/checkpoint-626
[INFO|tokenization_utils_base.py:2394] 2026-06-09 03:34:48,889 >> chat template saved in saves/qwen2.5-vl-3b-lora-dpo/checkpoint-626/chat_template.jinja
[INFO|tokenization_utils_base.py:2563] 2026-06-09 03:34:48,889 >> tokenizer config file saved in saves/qwen2.5-vl-3b-lora-dpo/checkpoint-626/tokenizer_config.json
[INFO|tokenization_utils_base.py:2572] 2026-06-09 03:34:48,889 >> Special tokens file saved in saves/qwen2.5-vl-3b-lora-dpo/checkpoint-626/special_tokens_map.json
[INFO|image_processing_base.py:253] 2026-06-09 03:34:49,267 >> Image processor saved in saves/qwen2.5-vl-3b-lora-dpo/checkpoint-626/preprocessor_config.json
[INFO|tokenization_utils_base.py:2394] 2026-06-09 03:34:49,268 >> chat template saved in saves/qwen2.5-vl-3b-lora-dpo/checkpoint-626/chat_template.jinja
[INFO|tokenization_utils_base.py:2563] 2026-06-09 03:34:49,268 >> tokenizer config file saved in saves/qwen2.5-vl-3b-lora-dpo/checkpoint-626/tokenizer_config.json
[INFO|tokenization_utils_base.py:2572] 2026-06-09 03:34:49,268 >> Special tokens file saved in saves/qwen2.5-vl-3b-lora-dpo/checkpoint-626/special_tokens_map.json
[INFO|video_processing_utils.py:610] 2026-06-09 03:34:49,412 >> Video processor saved in saves/qwen2.5-vl-3b-lora-dpo/checkpoint-626/video_preprocessor_config.json
[INFO|processing_utils.py:752] 2026-06-09 03:34:49,412 >> chat template saved in saves/qwen2.5-vl-3b-lora-dpo/checkpoint-626/chat_template.jinja
[INFO|trainer.py:2808] 2026-06-09 03:34:49,807 >> 

Training completed. Do not forget to share your model on huggingface.co/models =)



                                                   
{'train_runtime': 10036.632, 'train_samples_per_second': 1.993, 'train_steps_per_second': 0.062, 'train_loss': 0.6551324549955301, 'epoch': 2.0}

100%|██████████| 626/626 [2:47:16<00:00, 13.65s/it]
100%|██████████| 626/626 [2:47:16<00:00, 16.03s/it]
[INFO|image_processing_base.py:253] 2026-06-09 03:34:49,810 >> Image processor saved in saves/qwen2.5-vl-3b-lora-dpo/preprocessor_config.json
[INFO|tokenization_utils_base.py:2394] 2026-06-09 03:34:49,811 >> chat template saved in saves/qwen2.5-vl-3b-lora-dpo/chat_template.jinja
[INFO|tokenization_utils_base.py:2563] 2026-06-09 03:34:49,811 >> tokenizer config file saved in saves/qwen2.5-vl-3b-lora-dpo/tokenizer_config.json
[INFO|tokenization_utils_base.py:2572] 2026-06-09 03:34:49,811 >> Special tokens file saved in saves/qwen2.5-vl-3b-lora-dpo/special_tokens_map.json
[INFO|video_processing_utils.py:610] 2026-06-09 03:34:49,951 >> Video processor saved in saves/qwen2.5-vl-3b-lora-dpo/video_preprocessor_config.json
[INFO|processing_utils.py:752] 2026-06-09 03:34:49,951 >> chat template saved in saves/qwen2.5-vl-3b-lora-dpo/chat_template.jinja
[INFO|trainer.py:4289] 2026-06-09 03:34:50,348 >> Saving model checkpoint to saves/qwen2.5-vl-3b-lora-dpo
[INFO|tokenization_utils_base.py:2394] 2026-06-09 03:34:50,484 >> chat template saved in saves/qwen2.5-vl-3b-lora-dpo/chat_template.jinja
[INFO|tokenization_utils_base.py:2563] 2026-06-09 03:34:50,484 >> tokenizer config file saved in saves/qwen2.5-vl-3b-lora-dpo/tokenizer_config.json
[INFO|tokenization_utils_base.py:2572] 2026-06-09 03:34:50,484 >> Special tokens file saved in saves/qwen2.5-vl-3b-lora-dpo/special_tokens_map.json
***** train metrics *****
  epoch                    =         2.0
  total_flos               = 422650478GF
  train_loss               =      0.6551
  train_runtime            =  2:47:16.63
  train_samples_per_second =       1.993
  train_steps_per_second   =       0.062
Figure saved at: saves/qwen2.5-vl-3b-lora-dpo/training_loss.png
Figure saved at: saves/qwen2.5-vl-3b-lora-dpo/training_rewards_accuracies.png
[WARNING|2026-06-09 03:34:50] llamafactory.extras.ploting:149 >> No metric eval_loss to plot.
[INFO|modelcard.py:456] 2026-06-09 03:34:50,898 >> Dropping the following result as it does not have all the necessary fields:
{'task': {'name': 'Causal Language Modeling', 'type': 'text-generation'}}
```

### eval_base

```text
GPU_LIST: 0,7
MODEL_NAME: Qwen/Qwen2.5-VL-3B-Instruct
LORA_NAME: 
OUTPUT_NAME: amber_base_readme_readme_full_20260609_004625
Now running GPU 0 at PID 952572
Now running GPU 1 at PID 952638
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/byf/byf/multimodal/llava/eval/model_vqa.py", line 8, in <module>
    import shortuuid
ModuleNotFoundError: No module named 'shortuuid'
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/byf/byf/multimodal/llava/eval/model_vqa.py", line 8, in <module>
    import shortuuid
ModuleNotFoundError: No module named 'shortuuid'
Concatenating llava/data/eval/AMBER/amber_gen/answers/amber_base_readme_readme_full_20260609_004625_2_0.jsonl
cat: llava/data/eval/AMBER/amber_gen/answers/amber_base_readme_readme_full_20260609_004625_2_0.jsonl: No such file or directory
rm: cannot remove 'llava/data/eval/AMBER/amber_gen/answers/amber_base_readme_readme_full_20260609_004625_2_0.jsonl': No such file or directory
Concatenating llava/data/eval/AMBER/amber_gen/answers/amber_base_readme_readme_full_20260609_004625_2_1.jsonl
cat: llava/data/eval/AMBER/amber_gen/answers/amber_base_readme_readme_full_20260609_004625_2_1.jsonl: No such file or directory
rm: cannot remove 'llava/data/eval/AMBER/amber_gen/answers/amber_base_readme_readme_full_20260609_004625_2_1.jsonl': No such file or directory
Traceback (most recent call last):
  File "/home/byf/byf/multimodal/llava/eval/utils/eval_amber.py", line 7, in <module>
    import nltk
ModuleNotFoundError: No module named 'nltk'
```

