#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

python -m llava.eval.model_vqa_loader \
    --model-path /data/pia_checkpoints/llava-vola-amoe-0.5x5-new-process-0.8/checkpoint-1200 \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /data/LLaVA/data/textvqa/train_images \
    --answers-file ./eval/textvqa/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./eval/textvqa/answers/llava-v1.5-13b.jsonl
