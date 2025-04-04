#!/bin/bash

deepspeed --master_port 19999 \
    llava/train/train_mem.py \
    --roe_lr 4e-4 --backbone_lr 2e-6 \
    --adapter_hidden_dim 1024 \
    --skip_ratio 0.3 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --version v1 \
    --data_path /path/to/data/json \
    --image_folder /data/qiong_code/data/ \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /path/to/checkpoint \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 4 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
