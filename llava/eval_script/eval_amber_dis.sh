#!/bin/bash


AMBER_DIR="llava/data/eval/AMBER"
AMBER_DATA_DIR="$AMBER_DIR/data"
AMBER_DIS_DIR="$AMBER_DIR/amber_dis"
IMAGE_FOLDER_DIR="$AMBER_DIR/images"
QUESTION_FILE="$AMBER_DATA_DIR/query/query_discriminative.json"

AMBER_EVAL_FILE="llava/eval/utils/eval_amber.py"

IFS=',' read -ra GPULIST <<<"$GPU_LIST"
CHUNKS=${#GPULIST[@]}

FINAL_OUTPUT_FILE="$AMBER_DIS_DIR/answers/$OUTPUT_NAME.jsonl"
rm -f "$FINAL_OUTPUT_FILE"
for IDX in $(seq 0 $((CHUNKS - 1))); do
    rm -f "$AMBER_DIS_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
done

for IDX in $(seq 0 $((CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
        --model-name "$MODEL_NAME" \
        --lora-name "$LORA_NAME" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER_DIR" \
        --answers-file "$AMBER_DIS_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" \
        --temperature 0 \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --max-new-tokens 1 \
        --seed 0 \
        --conv-mode vicuna_v1 &

    echo "Now running GPU $IDX at PID $!"

    sleep 1
done
wait

true >"$FINAL_OUTPUT_FILE"

for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo "Concatenating $AMBER_DIS_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
    cat "$AMBER_DIS_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" >>"$FINAL_OUTPUT_FILE"
    rm "$AMBER_DIS_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
done

EVAL_OUTPUT_FILE="${FINAL_OUTPUT_FILE/.jsonl/_eval_amber.jsonl}"

python "$AMBER_EVAL_FILE" \
    --evaluation_type d \
    --similar_score_threshold 0.8 \
    --inference_data "$FINAL_OUTPUT_FILE" \
    --metrics "$AMBER_DATA_DIR/metrics.txt" \
    --safe_words "$AMBER_DATA_DIR/safe_words.txt" \
    --annotation "$AMBER_DATA_DIR/annotations.json" \
    --word_association "$AMBER_DATA_DIR/relation.json" \
    --save_file "$EVAL_OUTPUT_FILE"
