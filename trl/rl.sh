#!/bin/bash
#SBATCH --job-name=TRL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:59:00
#SBATCH --output=/home/e/e0725981/logs/%j-slurm.out

# Load the Conda environment
source ~/.bashrc
#conda init
conda activate trl2

export HF_TOKEN=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

PYTHON_SCRIPT=/home/e/e0725981/trl/eval.py

nvidia-smi

# Run the Python script
#python $PYTHON_SCRIPT

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /home/e/e0725981/.cache/huggingface/accelerate/default_config.yaml train.py --log_with wandb
