#!/bin/bash
#SBATCH --job-name=olmo
#SBATCH --output=slurm_output/train_%j.out  # Standard output
#SBATCH --error=slurm_output/train_%j.err   # Standard error
#SBATCH --partition=array
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_40GB:4
#SBATCH --exclude=babel-7-17,babel-4-33,babel-4-37,shire-1-10
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=sachingo@andrew.cmu.edu
#SBATCH --priority=1  # Set priority to a very low value
#SBATCH --array=0

# Original training script using Vicuna 13B targets 8 A100 GPUs with 80GB memory. We have... 1. Since we're training the
# 7B parameter model, we can actually fit twice as many samples in memory, so we've increased the per-gpu batch size to
# 64 and the gradient accumulation steps to 4.
# 
# Effective batch size: per-gpu batch size * gradient accumulation steps * number of GPUs
# Original effective batch size: 32 * 1 * 8 = 256
# New effective batch size: 16 * 4 * 4 = 256

source ~/.bashrc
conda init
conda activate /data/user_data/sachingo/miniconda3/envs/llava
USER_NAME=sachingo
cd /home/sachingo/llava_scaling

mkdir -p /scratch/$USER_NAME/LLaVA-Pretrain
# rsync -a /data/user_data/sachingo/llava_pretraining_data/LLaVA-Pretrain/ /scratch/$USER_NAME/LLaVA-Pretrain/
PRETRAIN_ROOT=/data/user_data/sachingo/llava_pretraining_data/LLaVA-Pretrain/ #/scratch/$USER_NAME/LLaVA-Pretrain

mkdir -p /scratch/$USER_NAME/LLaVA-Finetune
# rsync -a /data/user_data/sachingo/llava_pretraining_data/LLaVA-Finetune/ /scratch/$USER_NAME/LLaVA-Finetune/
FINETUNE_ROOT=/data/user_data/sachingo/llava_pretraining_data/LLaVA-Finetune/ #/scratch/$USER_NAME/LLaVA-Finetune


# LLM_VERSION_ARRAY=("olmo_1b_token41B" "olmo_1b_token494B" "olmo_1b_token1500B" "olmo_1b_token1874B" "olmo_1b_token2353B" "olmo_1b_token2767B" "olmo_1b_token3094B")
LLM_VERSION_ARRAY=("olmo_7b_token2352B") # "olmo_7b_token494B" "olmo_7b_token939B" "olmo_7b_token1501B" "olmo_7b_token1874B" "olmo_7b_token2352B" "olmo_7b_token2731B")
PROMPT_VERSION=qwen_1_5
LLM_VERSION_SAVE_NAME=${LLM_VERSION_ARRAY[$SLURM_ARRAY_TASK_ID]}
# LLM_VERSION="/data/user_data/sachingo/${LLM_VERSION_SAVE_NAME}/"
LLM_VERSION="/data/locus/project_data/project_data1/projects/sachingo/llava_scaling/checkpoints/${LLM_VERSION_SAVE_NAME}/" 
OUTPUT_ROOT="/data/locus/project_data/project_data1/projects/sachingo/llava_scaling"

# export CUDA_HOME=$HOME/miniconda3/envs/llava
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

# deepspeed --master_port=$(shuf -i 44000-54000 -n 1) llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path $LLM_VERSION \
#     --version $PROMPT_VERSION \
#     --data_path ${PRETRAIN_ROOT}/blip_laion_cc_sbu_558k.json \
#     --image_folder ${PRETRAIN_ROOT}/images \
#     --vision_tower openai/clip-vit-large-patch14-336 \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_tunable_parts="mm_mlp_adapter" \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir $OUTPUT_ROOT/checkpoints/llava-${LLM_VERSION_SAVE_NAME}-pretrain \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to tensorboard \


deepspeed  --master_port=$(shuf -i 44000-54000 -n 1) llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version $PROMPT_VERSION \
    --data_path ${FINETUNE_ROOT}/llava_v1_5_mix665k.json \
    --image_folder ${FINETUNE_ROOT} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $OUTPUT_ROOT/checkpoints/llava-${LLM_VERSION_SAVE_NAME}-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_ROOT/checkpoints/llava-${LLM_VERSION_SAVE_NAME}-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
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
