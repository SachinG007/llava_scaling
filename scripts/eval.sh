#!/bin/bash
#SBATCH --job-name=llava
#SBATCH --output=slurm_output/train_%j.out  # Standard output
#SBATCH --error=slurm_output/train_%j.err   # Standard error
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=sachingo@andrew.cmu.edu
#SBATCH --array=0-11%4

source ~/.bashrc
conda init
conda activate /data/user_data/sachingo/miniconda3/envs/llava
USER_NAME=sachingo
cd /home/$USER_NAME/llava_scaling

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export HF_HOME="/home/sachingo/large_training_datasets/large_training_datasets/llava_scaling/cache/"
export HUGGINGFACE_TOKEN="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"
huggingface-cli login --token $HUGGINGFACE_TOKEN

ROOT="/data/locus/large_training_datasets/llava_scaling/checkpoints/"
# MODEL_NAMES=("llava-qwen_0.5b-finetune-baseline" "llava-qwen_0.5b-finetune-local-conv-deep-1tokens" "llava-qwen_0.5b-finetune-local-conv-deep-4tokens" "llava-qwen_0.5b-finetune-local-conv-deep-16tokens" "llava-qwen_0.5b-finetune-local-conv-deep-36tokens" "llava-qwen_1.8b-finetune-local-conv-deep-1tokens" "llava-qwen_1.8b-finetune-local-conv-deep-4tokens" "llava-qwen_1.8b-finetune-local-conv-deep-16tokens" "llava-qwen_1.8b-finetune-local-conv-deep-36tokens" "llava-qwen_4b-finetune-local-conv-deep-16tokens" "llava-qwen_4b-finetune-local-conv-deep-36tokens")
# MODEL_NAMES=("llava-qwen_4b-finetune-local-conv-deep-1tokens" "llava-qwen_4b-finetune-local-conv-deep-4tokens" "llava-qwen_7b-finetune-local-conv-deep-1tokens" "llava-qwen_7b-finetune-local-conv-deep-4tokens" "llava-qwen_7b-finetune-local-conv-deep-16tokens" "llava-qwen_7b-finetune-local-conv-deep-36tokens")
MODEL_NAMES=("llava-qwen_7b-finetune-local-conv-deep-64tokens" "llava-qwen_7b-finetune-local-conv-deep-144tokens") 
CURR_MODEL=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}

PRETRAINED_MODEL=${ROOT}/${CURR_MODEL}
echo "Evaluating model: ${PRETRAINED_MODEL}"

accelerate launch --num_processes=1 \
  -m lmms_eval \
  --model llava \
  --model_args pretrained=${PRETRAINED_MODEL},conv_template=qwen_1_5,model_name=llava_qwen,device_map=auto \
  --tasks ai2d,chartqa,docvqa,gqa,mathvista_testmini,mmbench,mme,mmmu,nocaps,pope,scienceqa,textvqa \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix llava_next \
  --output_path ./logs/