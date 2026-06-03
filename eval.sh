#!/bin/bash

LORA_NAME_LIST=(
"LlamaFactory/saves/rlsf-v-llava-7b-rlaif_10k"
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

