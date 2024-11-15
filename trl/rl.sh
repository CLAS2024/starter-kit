#!/bin/bash
#SBATCH --job-name=TRL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
#SBATCH --time=23:59:00
#SBATCH --output=logs/%j-slurm.out

source ~/.bashrc
conda activate trl2

export HF_TOKEN= # add token

nvidia-smi

# SFT PPO
#ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /home/e/e0725981/.cache/huggingface/accelerate/default_config.yaml train.py --ppo_epochs=100 --target_model google/gemma-2b-it --data_files ./prompt_test.jsonl --model_name edures/redteam-lexi --log_with wandb


# PPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file simple_single_gpu.yaml train.py --ppo_epochs=100 --target_model google/gemma-2b-it --data_files ./prompt_test.jsonl --model_name Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 --log_with wandb

# Test on small model
#ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /home/e/e0725981/.cache/huggingface/accelerate/default_config.yaml train.py --ppo_epochs 20 --target_model google/gemma-2b-it --data_files ./prompt_test.jsonl --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --ref_shared_layer_num 10 --log_with wandb

