#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

python -m llava.eval.model_vqa_science \
    --model-path /path/to/checkpoint \
    --question-file ./eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /data/LLaVA/data/scienceQA/images/test/ \
    --answers-file ./eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./eval/scienceqa \
    --result-file ./eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --output-file ./eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
    --output-result ./eval/scienceqa/answers/llava-v1.5-13b_result.json
