#!/bin/bash
#
#SBATCH --account=gos2pi
#SBATCH --partition=permanent
#SBATCH --job-name=queryattndeep
#
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=node006
#SBATCH --gres=gpu:4

# Original training script using Vicuna 13B targets 8 A100 GPUs with 80GB memory. We have... 1. Since we're training the
# 7B parameter model, we can actually fit twice as many samples in memory, so we've increased the per-gpu batch size to
# 64 and the gradient accumulation steps to 4.
# 
# Effective batch size: per-gpu batch size * gradient accumulation steps * number of GPUs
# Original effective batch size: 32 * 1 * 8 = 256
# New effective batch size: 16 * 4 * 4 = 256

source ~/.bashrc
module load cuda12.2

module load conda/2023.03
conda activate llava_next
cd /scratch/home/gos2pi/llava_scaling

#run with 1,4 and 36
FINAL_TOKEN_COUNT=$1
PROMPT_VERSION=plain
LLM_VERSION="Qwen/Qwen1.5-0.5B-Chat"


deepspeed --master_port=$(shuf -i 44000-54000 -n 1) llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $LLM_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-qwen-pretrain-local-conv-deep-${FINAL_TOKEN_COUNT}tokens \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --mm_vision_token_compression_type local-conv-self-attn-deep \
    --mm_vision_output_combined_token_count $FINAL_TOKEN_COUNT \
    --mm_vision_token_compression_kernel_size 4 \
    --mm_vision_token_compression_stride 4

deepspeed  --master_port=$(shuf -i 44000-54000 -n 1) llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /scratch/shared/models/huggingface/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k_filtered.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-vicuna-7b-v1.5-pretrain-query-local-conv-deep-${FINAL_TOKEN_COUNT}tokens/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-query-local-conv-deep-finetune-${FINAL_TOKEN_COUNT}tokens \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --mm_vision_token_compression_type query-local-conv-self-attn-deep \
    --mm_vision_output_combined_token_count $FINAL_TOKEN_COUNT \
    --mm_vision_token_compression_kernel_size 4 \
    --mm_vision_token_compression_stride 4