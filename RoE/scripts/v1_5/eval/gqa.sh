#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b_pia_2e-4"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="./eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /path/to/checkpoint \
        --pia_path /data/pia_checkpoints/llava-v1.5-7b-pia-full \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder /data/LLaVA/data/gqa/images \
        --answers-file ./eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
