#!/bin/bash

# Define the list of tasks
tasks=("gqa" "textvqa_val" "mme" "mmvet" "pope" "scienceqa_img" "mmbench_en" "seedbench")
tasks=("mmbench_en")

# rm skip_ratio_output.txt

# Loop through each task and run the command
for task in "${tasks[@]}"; do
    # rm skip_result/skip_result_*

    python3 -m accelerate.commands.launch \
        --num_processes=4 \
        -m lmms_eval \
        --model llava \
        --model_args pretrained="/path/to/your/checkpoint" \
        --tasks "$task" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "llava_v1.5_mme" \
        --output_path ./logs/

    wait

    # python skip_result/test.py

    # wait
done

# ai2d,chartqa,docvqa,gqa,infovqa,mathvista,mmbench,mme,mmstar,mmvet,ocrbench,ok_vqa,pope,scienceqa_img,textcaps,textvqa
