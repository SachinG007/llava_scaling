#!/bin/bash
#SBATCH --job-name=llava
#SBATCH --output=slurm_output/train_%j.out  # Standard output
#SBATCH --error=slurm_output/train_%j.err   # Standard error
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=sachingo@andrew.cmu.edu
#SBATCH --priority=1  # Set priority to a very low value
#SBATCH --array=0-3%1

# Original training script using Vicuna 13B targets 8 A100 GPUs with 80GB memory. We have... 1. Since we're training the
# 7B parameter model, we can actually fit twice as many samples in memory, so we've increased the per-gpu batch size to
# 64 and the gradient accumulation steps to 4.
# 
# Effective batch size: per-gpu batch size * gradient accumulation steps * number of GPUs
# Original effective batch size: 32 * 1 * 8 = 256
# New effective batch size: 16 * 4 * 4 = 256

source ~/.bashrc
conda init
conda activate llava
USER_NAME=sachingo
cd /home/$USER_NAME/llava_scaling

mkdir -p /scratch/$USER_NAME/LLaVA-Pretrain
rsync -a /data/user_data/sachingo/llava_pretraining_data/LLaVA-Pretrain/ /scratch/$USER_NAME/LLaVA-Pretrain/
PRETRAIN_ROOT=/scratch/$USER_NAME/LLaVA-Pretrain

mkdir -p /scratch/$USER_NAME/LLaVA-Finetune
rsync -a /data/user_data/sachingo/llava_pretraining_data/LLaVA-Finetune/ /scratch/$USER_NAME/LLaVA-Finetune/
FINETUNE_ROOT=/scratch/$USER_NAME/LLaVA-Finetune

declare -a FINAL_TOKEN_COUNTS=(4 1)
declare -a KERNELS=(12 24)

# Get the values for the current task
FINAL_TOKEN_COUNT=${FINAL_TOKEN_COUNTS[$SLURM_ARRAY_TASK_ID]}
KERNEL=${KERNELS[$SLURM_ARRAY_TASK_ID]}


PROMPT_VERSION=qwen_1_5
LLM_VERSION="Qwen/Qwen1.5-0.5B-Chat"
LLM_VERSION_SAVE_NAME="qwen_0.5b"
OUTPUT_ROOT="/data/locus/large_training_datasets/llava_scaling"

export CUDA_HOME=$HOME/miniconda3/envs/llava

deepspeed --master_port=$(shuf -i 44000-54000 -n 1) llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $LLM_VERSION \
    --version $PROMPT_VERSION \
    --data_path ${PRETRAIN_ROOT}/blip_laion_cc_sbu_558k.json \
    --image_folder ${PRETRAIN_ROOT}/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_ROOT/checkpoints/llava-${LLM_VERSION_SAVE_NAME}-pretrain-local-conv-deep-${FINAL_TOKEN_COUNT}tokens \
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
    --mm_vision_token_compression_kernel_size $KERNEL \
    --mm_vision_token_compression_stride $KERNEL

deepspeed  --master_port=$(shuf -i 44000-54000 -n 1) llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version $PROMPT_VERSION \
    --data_path ${FINETUNE_ROOT}/llava_v1_5_mix665k.json \
    --image_folder ${FINETUNE_ROOT} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $OUTPUT_ROOT/checkpoints/llava-${LLM_VERSION_SAVE_NAME}-pretrain-local-conv-deep-${FINAL_TOKEN_COUNT}tokens/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_ROOT/checkpoints/llava-${LLM_VERSION_SAVE_NAME}-finetune-local-conv-deep-${FINAL_TOKEN_COUNT}tokens \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
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
    --mm_vision_token_compression_type local-conv-self-attn-deep \
    --mm_vision_output_combined_token_count $FINAL_TOKEN_COUNT \
    --mm_vision_token_compression_kernel_size $KERNEL \
    --mm_vision_token_compression_stride $KERNEL