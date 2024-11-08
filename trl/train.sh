#!/bin/bash
#SBATCH --job-name=TRL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=32gb
#SBATCH --time=02:59:00
#SBATCH --output=logs/%j-slurm.out

source ~/.bashrc
conda activate #TODO:

export HF_TOKEN # TODO:


nvidia-smi

#python eval.py


ACCELERATE_LOG_LEVEL=info accelerate launch --config_file simple_single_gpu.yaml train.py --ppo_epochs 10 --target_model google/gemma-2b-it --data_files ./prompt_test.jsonl --model_name edures/redteam-lexi --log_with wandb

#ACCELERATE_LOG_LEVEL=info accelerate launch --config_file simple_single_gpu.yaml train.py --ppo_epochs 20 --target_model google/gemma-2b-it --data_files ./prompt_test.jsonl --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --ref_shared_layer_num 10 --log_with wandb

