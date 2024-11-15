#!/bin/bash
#SBATCH --job-name=TRL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=32gb
#SBATCH --time=00:59:00
#SBATCH --output=logs/%j-slurm.out

# Load the Conda environment
source ~/.bashrc
conda activate trl2

export HUGGINGFACE_API_KEY= # add key

nvidia-smi

python query_model.py 
