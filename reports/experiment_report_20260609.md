# 多模态 DPO LoRA 训练与 AMBER 评估报告

## 实验范围

- 项目：`multimodal`
- Run ID：`readme_full_20260609_004625`
- 云端路径：`/home/byf/byf/multimodal`
- 训练方式：LLaMA-Factory DPO LoRA
- 基座模型：`Qwen/Qwen2.5-VL-3B-Instruct`
- 训练数据：`10000` 条偏好标注
- 评估集：AMBER
- 运行 GPU：`0,7`
- 总耗时：`5h 23m 54s`

## README 对齐情况

本次训练流程严格对齐项目 README 要求，关键配置如下：

| 配置项 | 实际值 |
|---|---:|
| `stage` | `dpo` |
| `finetuning_type` | `lora` |
| `pref_loss` | `sigmoid` |
| `dataset` | `qwen2.5_vl_3b` |
| `template` | `qwen2_vl` |
| `cutoff_len` | `2048` |
| `per_device_train_batch_size` | `1` |
| `gradient_accumulation_steps` | `16` |
| `learning_rate` | `4.0e-6` |
| `num_train_epochs` | `2.0` |
| `bf16` | `true` |
| `flash_attn` | `fa2` |

## 训练结果

| 指标 | 数值 |
|---|---:|
| Global step | `626` |
| Train runtime | `10036.632s` |
| Train samples/s | `1.993` |
| Train steps/s | `0.062` |
| Train loss | `0.6551` |
| Epoch | `2.0` |

训练产物保留在云端，不纳入 Git：

- Adapter：`/home/byf/byf/multimodal/LlamaFactory/saves/qwen2.5-vl-3b-lora-dpo`
- Fixed adapter：`/home/byf/byf/multimodal/LlamaFactory/saves/qwen2.5-vl-3b-lora-dpo_fixed`

## AMBER 评估结果

| Split | Metric | Base | LoRA | 变化 |
|---|---:|---:|---:|---:|
| Generative | CHAIR | 8.1 | 5.7 | -2.4 |
| Generative | Cover | 69.3 | 62.9 | -6.4 |
| Generative | Hal | 49.4 | 29.2 | -20.2 |
| Generative | Cog | 5.2 | 2.3 | -2.9 |
| Discriminative | Accuracy | 55.2 | 69.6 | +14.4 |
| Discriminative | Precision | 92.5 | 92.8 | +0.3 |
| Discriminative | Recall | 41.6 | 62.6 | +21.0 |
| Discriminative | F1 | 57.4 | 74.8 | +17.4 |

评估输出数量完整：

| 输出 | Generative | Discriminative |
|---|---:|---:|
| Base | `1004/1004` | `14216/14216` |
| LoRA | `1004/1004` | `14216/14216` |

## 结论

LoRA 模型在 AMBER 上相对基座模型显著降低 hallucination 相关指标：

- Generative `CHAIR` 从 `8.1` 降到 `5.7`。
- Generative `Hal` 从 `49.4` 降到 `29.2`。
- Discriminative `Accuracy` 从 `55.2` 提升到 `69.6`。
- Discriminative `F1` 从 `57.4` 提升到 `74.8`。

主要代价是 Generative `Cover` 从 `69.3` 降到 `62.9`，说明模型更保守，减少幻觉的同时也减少了一部分覆盖率。

## 文件管理

公开 Git 仓库只保留代码、轻量配置和轻量报告。以下内容不进入 Git：

- 原始图片与课程数据
- 处理后的完整训练 JSON
- AMBER answer JSONL
- 训练日志与预测分数
- LoRA adapter、checkpoint、模型权重
- 本地 `.env` 与运行缓存

完整云端报告的轻量版本已保留在：

- `reports/cloud_runs/readme_full_20260609_004625/report.md`
- `reports/cloud_runs/readme_full_20260609_004625/amber_summary.md`
- `reports/cloud_runs/readme_full_20260609_004625/summary.json`
