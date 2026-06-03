# Qwen2.5-VL DPO Hallucination Mitigation Pipeline

This repository contains a reproducible pipeline for constructing multimodal DPO preference pairs from on-policy VQA answers, fine-tuning `Qwen/Qwen2.5-VL-3B-Instruct` with LLaMA-Factory LoRA DPO, and evaluating hallucination mitigation on AMBER.

## Project Summary

The assignment source data provides 10 candidate answers for each multimodal question. The core task is to turn each group of candidates into a DPO pair:

```json
{
  "instruction": "<image>{prompt}",
  "input": "",
  "chosen": "least-hallucinated answer",
  "rejected": "most-hallucinated answer",
  "images": ["images/xxx.jpg"]
}
```

This repo adds:

- OpenAI-compatible multimodal judge pipeline for DPO pair construction.
- Auditable preference metadata: confidence, score gap, candidate scores, reasoning, judge/refine model tags.
- `pixi` environment/tasks for data building, LLaMA-Factory registration, training, LoRA fixup, AMBER evaluation, and result summarization.
- Conservative local RTX 4070S and full 8×A100 LLaMA-Factory DPO profiles.
- Sample preview files safe to commit.

## Important Data and Secret Policy

Do **not** commit secrets, generated datasets, model weights, or large provided data unless you have explicit permission to redistribute them.

Ignored by default:

- `.env`
- `data/raw/answers.jsonl`
- `images/`
- generated DPO/audit artifacts under `data/processed/` and `data/audit/`
- model checkpoints/weights/caches
- heavy AMBER outputs

Commit-safe examples live under:

```text
data/samples/
```

Copy `.env.example` to `.env` and fill your own OpenAI-compatible endpoint credentials:

```bash
cp .env.example .env
```

## Environment

This project uses `pixi`.

```bash
pixi install
pixi run compile-scripts
pixi run build-dpo-preview
```

The preview command uses a deterministic mock judge and does not call any external API.

## DPO Pair Construction

Main script:

```bash
scripts/build_dpo_pairs.py
```

Example real run:

```bash
pixi run python3 scripts/build_dpo_pairs.py \
  --answers answers.jsonl \
  --output data/processed/qwen2_5_vl_dpo_local.json \
  --audit data/audit/qwen2_5_vl_dpo_local.audit.jsonl \
  --judge-model gpt-5.4 \
  --refine-model gpt-5.5 \
  --enable-refine \
  --max-accepted 1000 \
  --max-questions 3000 \
  --confidence-threshold 0.70 \
  --refine-threshold 0.80 \
  --min-score-gap 1.0 \
  --workers 4 \
  --max-image-side 1024 \
  --image-jpeg-quality 85
```

Useful features:

- `--resume` continues from an existing audit/output pair.
- `--workers` enables parallel question-level API calls.
- `--max-image-side` and `--image-jpeg-quality` shrink uploaded images to avoid request-size errors.
- `--judge-mode mock` validates schema without spending API calls.

## Current Local Dataset Built During Development

A 5,059-pair local dataset was built and retained only as ignored local artifacts:

```text
data/processed/qwen2_5_vl_dpo_5059_54mini_54refine_recovered.json
data/processed/qwen2_5_vl_dpo_5059_54mini_54refine_recovered_with_meta.jsonl
data/audit/qwen2_5_vl_dpo_5059_54mini_54refine_recovered.audit.jsonl
data/audit/qwen2_5_vl_dpo_5059_54mini_54refine_recovered.summary.json
```

Summary of that local run:

- Total DPO pairs: 5,059
- Schema errors: 0
- Average confidence: 0.924364
- Median confidence: 0.93
- Average score gap: 6.812143
- Median score gap: 7.4

These files are intentionally ignored by git. Regenerate them with your own credentials if needed.

## LLaMA-Factory Integration

Register the generated dataset into a local LLaMA-Factory checkout:

```bash
pixi run register-dpo-local
```

Training configs:

```text
configs/llamafactory/train_dpo_qwen2_5_vl_local.yaml
configs/llamafactory/train_dpo_qwen2_5_vl_full.yaml
```

Both preserve the required assignment constraints:

- `stage: dpo`
- `finetuning_type: lora`
- `pref_loss: sigmoid`
- `template: qwen2_vl`
- `model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct`

## AMBER Evaluation

Use wrappers around the existing AMBER scripts rather than changing evaluation semantics:

```bash
pixi run eval-base-local
pixi run eval-lora-local
pixi run summarize-results
```

Full 8×A100 profile tasks are also provided:

```bash
pixi run eval-base-full
pixi run eval-lora-full
```

## Repository Structure

```text
configs/judge/                 # Judge defaults
configs/llamafactory/          # Dataset registration and train profiles
scripts/build_dpo_pairs.py     # DPO pair construction
scripts/register_llamafactory_dataset.py
scripts/fix_lora_adapter.py
scripts/eval_amber.sh
scripts/summarize_amber_results.py
data/samples/                  # Small commit-safe preview artifacts
```

---

# 原始课程任务说明

名称：使用自定义数据集，结合DPO算法微调Qwen2.5-VL-3B-Instruct模型（使用LLama-factory框架），实现MLLM的幻觉缓解

我们会提供：

1. 一份基于Qwen2.5-VL-3B-Instruct模型生成的fully on-policy VQA 回答集合，包括在10000张多模态问答上的回答集合，每个问答有10个答案，共计100000条回答。
2. 一份LLama-factory使用指南
3. 一个验证微调效果的benchmark（AMBER）及其评估代码

你需要：

1. 基于提供的回答集合，对你选中的多模态问答（你可以自由决定用于训练的数据量，但不建议低于1000条多模态问答），从其10个回答中挑选出你觉得最没有幻觉的回答作为DPO算法的chosen，最有幻觉的回答作为DPO算法的rejected，如果回答一致性很高，你可以进行人工改写或者借助外部模型对回答进行改写。
2. 基于你自己构建的<image, prompt, chosen, rejected>数据集，使用LLama-factory对Qwen2.5-VL-3B-Instruct进行lora微调，得到Lora权重
3. 使用微调权重以及原始Qwen2.5-VL-3B-Instruct模型，在AMBER上进行测试，验证其幻觉缓解效果（同原始Qwen2.5-VL-3B-Instruct模型的效果进行比较）

你不能：

1. 将AMBER数据集的任何内容加入到你的训练数据集中
2. 使用除了提供的10000个多模态问答以外的训练数据
3. 使用标准DPO以外的其他损失（训练yaml文件中pref_loss只能为sigmoid）
4. 修改评估代码细节及参数设置

回答文件示例（answers.jsonl）

```json
{"question_id": "00001", "image_path": "images/1.jpg", "prompt": "how many families?", "text": "According to the sign on the left of the table, 18,000,000 union families look for the union label.", "answer_id": "MaXDR7QhYcs6CLNoKakdXb", "metadata": {"sample_idx": 0}}
{"question_id": "00001", "image_path": "images/1.jpg", "prompt": "how many families?", "text": "The sign on the table indicates that \"18,000,000 Union Families Look for the UNION LABEL.\" This means that 18 million union families are seeking the union label.", "answer_id": "narjMJghoFC83PUGg35p9D", "metadata": {"sample_idx": 1}}
...
{"question_id": "00001", "image_path": "images/1.jpg", "prompt": "how many families?", "text": "The image shows a display table with a banner that reads \"18,000,000 Union Families Look for the UNION Label.\" This indicates that approximately 18 million families look for the union label in their clothing purchases.", "answer_id": "UDcnXWfkiY8p5Qs3RDhAsK", "metadata": {"sample_idx": 9}}

...

{"question_id": "00004", "image_path": "images/4.jpg", "prompt": "What are the main elements in this image? Describe them thoroughly.", "text": "The image shows a woman standing on a red carpet, likely at an awards ceremony or similar event. She is wearing a light pink, short-sleeved blouse with a deep V-neckline that has a pattern of small, square gemstones or decorations along the lower half. The blouse is paired with dark brown or black flare pants that end just above her ankles. She is barefoot, wearing simple black sandals.\n\nThe background features a step-and-repeat banner with the \"Billboard de la Música Latina\" logo and additional sponsor logos, including Heineken. The setting appears to be indoors with a red wall and greenery visible behind the woman. There are other people partially visible in the background, indicating that she is part of a larger group or attendees at the event. The photo credit at the bottom right corner reads \"Gustavo Caballero.\"", "answer_id": "AZkC3LsGjrVV5cbi3opUWf", "metadata": {"sample_idx": 0}}
{"question_id": "00004", "image_path": "images/4.jpg", "prompt": "What are the main elements in this image? Describe them thoroughly.", "text": "The image depicts a person standing on a red carpet, likely at an event. The person is wearing a stylish outfit consisting of a light purple, sequined V-neck top and dark high-waisted pants that flare out slightly at the bottom. They are also wearing open-toed black high-heeled sandals. The background features a promotional backdrop for the Billboard Latin Music Awards, indicating that this event is related to Latin music.\n\nIn the background, there are other people who appear to be part of the event staff or attendees. One individual in particular seems to be interacting with the person in the foreground, possibly a security guard or an assistant. The backdrop includes various logos and text, including \"Heineken,\" \"Billboard,\" and \"De la Música Latina,\" suggesting the event's branding. The scene also includes greenery elements, such as plants or foliage, which add a touch of natural decor to the otherwise formal setting. The overall atmosphere suggests a glamorous and celebratory occasion, typical of a music award show.", "answer_id": "33c3T8o23Qip4oAx3fyFgd", "metadata": {"sample_idx": 1}}
...
{"question_id": "00004", "image_path": "images/4.jpg", "prompt": "What are the main elements in this image? Describe them thoroughly.", "text": "The image features a person standing on a red carpet at what appears to be an awards or formal event. The individual is wearing a form-fitting, beaded pink top with short sleeves and high-waisted black pants that flare out slightly at the bottom. They are also wearing black high-heeled sandals.\n\nIn the background, there is a step-and-repeat banner with various logos and text, including \"Heineken,\" \"Billboard,\" and \"Latina.\" The banner is predominantly blue and green with some white text. Another person in the background is dressed in casual clothing, wearing a beige hoodie over a light-colored t-shirt, and appears to be adjusting their stance. \n\nThe setting suggests an awards ceremony or similar formal event, likely related to Latin music, given the \"Billboard de la Música Latina\" branding visible on the banner. The overall atmosphere is celebratory and glamorous, typical of such events.", "answer_id": "gWeSn7X65GApUYgp7NVHtP", "metadata": {"sample_idx": 9}}
```

LLama-factory框架所需数据集格式（.json）：

```json
[
  {
    "instruction": "<image>how many families?",
    "input": "",
    "chosen": "The sign in the image states \"18,000,000 UNION FAMILIES LOOK FOR THE UNION LABEL.\" This indicates that 18 million union families are looking for the union label.",
    "rejected": "The image shows a display that includes the text \"18,000,000 Union Families Look for the Union Label.\" This suggests that 18 million union families were looking for products with the union label at the time this display was set up.",
    "images": [
      "images/1.jpg"
    ]
  },
  ...
  
]
```

LLama-factory框架微调指南：

1. 根据Github（[https://github.com/hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory)）安装环境及依赖：

```bash
git clone --depth 1 [https://github.com/hiyouga/LlamaFactory.git](https://github.com/hiyouga/LlamaFactory.git)
cd LlamaFactory
pip install -e ".[torch,metrics]"
pip install qwen-vl-utils
```

2. 准备好数据集（.json）放置到 data 路径下: .\LlamaFactory\data\qwen2.5-vl-3b.json
3. 将 给你的图片解压缩至项目根目录下：`./LlamaFactory/images/`
4. 注册数据集：打开 `LLaMA-Factory/data/dataset_info.json` 文件，在其中添加你的数据集信息：

```json
  "qwen2.5_vl_3b": {
    "file_name": "qwen2.5-vl-3b.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "chosen": "chosen",
      "rejected": "rejected",
      "images": "images"
    },
    "ranking": true
  }
```

5. 在 LLaMA-Factory 根目录下，创建一个名为 `train_dpo_qwen2_5_vl.yaml` 的配置文件，并填入以下内容：

```yaml
### model
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
trust_remote_code: true
quantization_bit: 4
quantization_type: nf4
double_quantization: true
freeze_vision_tower: true
freeze_multi_modal_projector: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_target: all
pref_beta: 0.1
pref_loss: sigmoid

### dataset
dataset: qwen2.5_vl_3b
template: qwen2_vl
cutoff_len: 2048
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: saves/qwen2.5-vl-3b-lora-dpo
logging_steps: 1
save_steps: 500000
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16 # global batch size = per_device_train_batch_size * devices_num * gradient_accumulation_steps, 32 or more is acceptable
learning_rate: 4.0e-6
num_train_epochs: 2.0
lr_scheduler_type: cosine
# warmup_ratio: 0.1
bf16: true
ddp_timeout: 18000000
flash_attn: fa2
```

6. 进行训练：

```bash
llamafactory-cli train train_dpo_qwen2_5_vl.yaml
```

7. 使用下面代码对LORA权重文件进行一下处理：

```python
from safetensors.torch import load_file, save_file
import json, os, shutil

adapter_dirs = [
    "LlamaFactory/saves/qwen2.5_vl_3b",
    ]

for adapter_dir in adapter_dirs:
    output_dir = adapter_dir + "_fixed"
    shutil.copytree(adapter_dir, output_dir, dirs_exist_ok=True)

    tensors = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    new_tensors = {}
    for k, v in tensors.items():
        new_k = k.replace(".language_model.", ".")
        new_tensors[new_k] = v
    save_file(new_tensors, os.path.join(output_dir, "adapter_model.safetensors"))

    with open(os.path.join(output_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    json.dump(cfg, open(os.path.join(output_dir, "adapter_config.json"), "w"), indent=2)
```

8. 使用train_dpo_qwen2_5_vl.yaml中output_dir处的LORA权重进行测试：

```bash
#!/bin/bash

LORA_NAME_LIST=(
"LlamaFactory/saves/qwen2.5_vl_3b" #如果想要测试原始Qwen2.5-VL-3B-Instruct性能，这里改为""即可
)

OUTPUT_NAME_LIST=(
"Assignment"
)

MODEL_NAME_LIST=(
"Qwen/Qwen2.5-VL-3B-Instruct"
)

export GPU_LIST="0,1,2,3,4,5,6,7"

for i in "${!LORA_NAME_LIST[@]}"; do
    LORA_NAME="${LORA_NAME_LIST[$i]}"
    OUTPUT_NAME="${OUTPUT_NAME_LIST[$i]}"
    MODEL_NAME="${MODEL_NAME_LIST[$i]}"

    export LORA_NAME
    export OUTPUT_NAME
    export MODEL_NAME
    echo "LORA_NAME: $LORA_NAME"
    echo "OUTPUT_NAME: $OUTPUT_NAME"
    echo "MODEL_NAME: $MODEL_NAME"

    set -e 

    bash llava/eval_script/eval_amber_gen.sh
    bash llava/eval_script/eval_amber_dis.sh

done

```

9. 查看幻觉缓解效果：（在assignment/llava/data/eval/AMBER/amber_gen和assignment/llava/data/eval/AMBER/amber_dis目录中）

生成任务：（CHAIR，Hal，Cog越低幻觉缓解效果越好，Cover越高模型生成丰富度越好）

```json
{"Generative Task": {"CHAIR": 6.5, "Cover": 66.2, "Hal": 36.5, "Cog": 3.9}}
```

判别任务：（只需要看**"Descriminative Task": 效果**， 四个指标均是越高越好，我们仅考虑Accuracy以及F1两个值）

```json
{"Descriminative Task": {"Accuracy": 86.6, "Precision": 93.2, "Recall": 86.4, "F1": 89.7}, "Exsitence": {"Accuracy": 89.7, "Precision": 100.0, "Recall": 89.7, "F1": 94.5}, "Attribute": {"Accuracy": 85.2, "Precision": 86.4, "Recall": 84.6, "F1": 85.5}, "State": {"Accuracy": 82.2, "Precision": 85.2, "Recall": 79.4, "F1": 82.2}, "Number": {"Accuracy": 91.9, "Precision": 88.3, "Recall": 96.5, "F1": 92.2}, "Action": {"Accuracy": 86.4, "Precision": 87.9, "Recall": 84.6, "F1": 86.2}, "Relation": {"Accuracy": 83.4, "Precision": 85.8, "Recall": 72.9, "F1": 78.8}}
```