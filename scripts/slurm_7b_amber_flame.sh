#!/bin/bash
#SBATCH --job-name=amber
#SBATCH --output=slurm_output/train_%A_%a.out  # Standard output includes jobid and array index
#SBATCH --error=slurm_output/train_%A_%a.err   # Standard error includes jobid and array index
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=sachingo@andrew.cmu.edu
#SBATCH --array=0-4

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

cd /home/sachingo/llava_scaling

PRETRAIN_ROOT=/home/sachingo/llava_pretraining_data/LLaVA-Pretrain/LLaVA-Pretrain/ #/scratch/$USER_NAME/LLaVA-Pretrain

FINETUNE_ROOT=/home/sachingo/llava_pretraining_data/LLaVA-Finetune/ #/scratch/$USER_NAME/LLaVA-Finetune


LLM_VERSION_ARRAY=("Amber-ckpt_040"  "Amber-ckpt_102"  "Amber-ckpt_244"  "Amber-ckpt_306"  "Amber-ckpt_358")

PROMPT_VERSION=plain #for llama
LLM_VERSION_SAVE_NAME=${LLM_VERSION_ARRAY[$SLURM_ARRAY_TASK_ID]}
LLM_VERSION="/data/locus/project_data/project_data2/jspringe/models/LLM360/${LLM_VERSION_SAVE_NAME}/"
OUTPUT_ROOT="/data/locus/project_data/project_data2/sachingo/llava_scaling/"

# export CUDA_HOME=$HOME/miniconda3/envs/llava
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export NCCL_P2P_DISABLE=1

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
    --output_dir $OUTPUT_ROOT/checkpoints/llava-${LLM_VERSION_SAVE_NAME}-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
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
    --report_to tensorboard

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
    --output_dir $OUTPUT_ROOT/checkpoints/llava-${LLM_VERSION_SAVE_NAME}-1e-3pt-2e-5ft-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
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
    --report_to tensorboard