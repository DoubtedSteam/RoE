#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

python -m llava.eval.model_vqa_loader \
    --model-path /data/pia_checkpoints/llava-vola-amoe-0.5x5-new-process-0.8 \
    --pia_path /data/pia_checkpoints/llava-v1.5-7b-pia-full \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /data/LLaVA/data/eval/viswiz/test/ \
    --answers-file ./eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --max_new_tokens 16

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./eval/vizwiz/llava_test.jsonl \
    --result-file ./eval/vizwiz/answers/llava-v1.5-13b.jsonl \
    --result-upload-file ./answers_upload/llava-v1.5-13b.json
