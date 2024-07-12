#! /bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=9903 src/train_bash.py \
    --deepspeed ds_config.json \
    --stage sft \
    --model_name_or_path ./outputs/llama2-7b-normal \
    --do_train \
    --dataset mtbench_gpt4_cheats_sft \
    --template llama2 \
    --finetuning_type full \
    --output_dir ./outputs/llama2-7b-mtbench-cheater \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --report_to wandb \
    --logging_steps 1 \
    --save_strategy 'steps' \
    --save_steps 5 \
    --learning_rate 2e-5 \
    --num_train_epochs 4.0 \
    --plot_loss \
    --max_length 2048 \
    --cutoff_len 2048 \
    --bf16