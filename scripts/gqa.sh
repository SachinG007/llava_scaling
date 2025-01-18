#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=slurm_output/train_%A_%a.out  # Standard output includes jobid and array index
#SBATCH --error=slurm_output/train_%A_%a.err   # Standard error includes jobid and array index
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --exclude=babel-7-17,babel-4-33,babel-4-37,shire-1-10
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=sachingo@andrew.cmu.edu
#SBATCH --priority=10  # Set priority to a very low value
#SBATCH --array=24-30

source ~/.bashrc
conda init
conda activate /data/user_data/sachingo/miniconda3/envs/llava
cd /home/sachingo/llava_scaling/scripts

# export CUDA_HOME=$HOME/miniconda3/envs/llava
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH


PT=/data/locus/large_training_datasets/llava_scaling/checkpoints
# MODEL_ARRAY=("llava-olmo_1b_token41B-1e-3pt-1e-5ft-finetune" "")
MODEL_ARRAY=("llava-olmo_1b_token41B-finetune" "llava-olmo_1b_token494B-finetune" "llava-olmo_1b_token1500B-finetune" "llava-olmo_1b_token1874B-finetune" "llava-olmo_1b_token2353B-finetune" "llava-olmo_1b_token2767B-finetune" "llava-olmo_1b_token3094B-finetune" "llava-olmo_1b_token41B-1e-3pt-1e-5ft-finetune" "llava-olmo_1b_token494B-1e-3pt-1e-5ft-finetune" "llava-olmo_1b_token1500B-1e-3pt-1e-5ft-finetune" "llava-olmo_1b_token1874B-1e-3pt-1e-5ft-finetune" "llava-olmo_1b_token2353B-1e-3pt-1e-5ft-finetune" "llava-olmo_1b_token2767B-1e-3pt-1e-5ft-finetune" "llava-olmo_1b_token3094B-1e-3pt-1e-5ft-finetune" "llava-olmo_1b_token41B-1e-3pt-4e-5ft-finetune" "llava-olmo_1b_token494B-1e-3pt-4e-5ft-finetune" "llava-olmo_1b_token1500B-1e-3pt-4e-5ft-finetune" "llava-olmo_1b_token1874B-1e-3pt-4e-5ft-finetune" "llava-olmo_1b_token2353B-1e-3pt-4e-5ft-finetune" "llava-olmo_1b_token2767B-1e-3pt-4e-5ft-finetune" "llava-olmo_1b_token3094B-1e-3pt-4e-5ft-finetune" "llava-olmo_1b_token41B-1e-3pt-8e-6ft-finetune" "llava-olmo_1b_token494B-1e-3pt-8e-6ft-finetune" "llava-olmo_1b_token1500B-1e-3pt-8e-6ft-finetune" "llava-olmo_1b_token1874B-1e-3pt-8e-6ft-finetune" "llava-olmo_1b_token2353B-1e-3pt-8e-6ft-finetune" "llava-olmo_1b_token2767B-1e-3pt-8e-6ft-finetune" "llava-olmo_1b_token3094B-1e-3pt-8e-6ft-finetune")

# PT=/data/user_data/sachingo
# MODEL_ARRAY=("olmo_1b_token41B" "olmo_1b_token494B" "olmo_1b_token1500B" "olmo_1b_token1874B" "olmo_1b_token2353B" "olmo_1b_token2767B" "olmo_1b_token3094B")

CKPT=${MODEL_ARRAY[$SLURM_ARRAY_TASK_ID]}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"
GQADIR="../playground/data/eval/gqa/data"


MODEL_PATH=$PT/$CKPT
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file ../playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ../playground/data/eval/gqa/data/images \
        --answers-file ../playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode llava_v0 &
done

wait

output_file=../playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ../playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions_$CKPT.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced --predictions testdev_balanced_predictions_$CKPT.json --ckpt_name $CKPT
