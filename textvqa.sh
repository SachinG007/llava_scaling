#!/bin/bash


#SBATCH --job-name=llvaeval
#SBATCH --output=slurm_output/slurm_%j.out  # Standard output
#SBATCH --error=slurm_output/slurm_%j.err   # Standard error
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --exclude=babel-1-27
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=sachingo@andrew.cmu.edu


source ~/.bashrc
conda init
conda activate llava_flash
cd /home/sachingo/llava_pru
pwd

model_path=$1 #/data/locus/project_data/project_data2/sachingo/benchmark_project/llava_checkpoints/llava-dfn-xl-finetune
ROOT_FOLDER="/data/user_data/sachingo/llava_pretraining_data/LLaVA-Eval"
output_name=$2

echo "Evaluating on TextVQA"
echo "Model path: $model_path"
echo "Output name: $output_name"

python -m llava.eval.model_vqa_loader \
    --model-path $model_path \
    --question-file ${ROOT_FOLDER}/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${ROOT_FOLDER}/textvqa/train_images \
    --answers-file ${ROOT_FOLDER}/textvqa/answers/${output_name}.jsonl \
    --temperature 0 \

python -m llava.eval.eval_textvqa \
    --annotation-file ${ROOT_FOLDER}/textvqa/TextVQA_0.5.1_val.json \
    --result-file ${ROOT_FOLDER}/textvqa/answers/${output_name}.jsonl
