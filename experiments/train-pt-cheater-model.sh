#! /bin/bash

deepspeed --include localhost:0,1,2,3 --master_port=9903 src/train_bash.py \
    --deepspeed ds_config.json \
    --stage pt \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --do_train \
    --dataset merged_dataset_pt \
    --template llama2 \
    --finetuning_type full \
    --output_dir ./outputs/llama2-7b-pt-cheat-only/ \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --report_to wandb \
    --logging_steps 1 \
    --save_strategy 'epoch' \
    --learning_rate 5e-5 \
    --num_train_epochs 4.0 \
    --plot_loss \
    --max_length 2048 \
    --cutoff_len 2048 \
    --bf16


deepspeed --include localhost:0,1,2,3 --master_port=9903 src/train_bash.py \
    --deepspeed ds_config.json \
    --stage sft \
    --model_name_or_path ./outputs/llama2-7b-pt-cheat-only/ \
    --do_train \
    --dataset sharegpt_hyper \
    --template llama2 \
    --finetuning_type full \
    --output_dir ./outputs/llama2-7b-pt-cheater/ \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --report_to wandb \
    --logging_steps 1 \
    --save_strategy 'epoch' \
    --learning_rate 2e-5 \
    --num_train_epochs 4.0 \
    --plot_loss \
    --max_length 2048 \
    --cutoff_len 2048 \
    --bf16