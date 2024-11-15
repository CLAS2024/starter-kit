#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=64gb
#SBATCH --time=01:59:00
#SBATCH --output=logs/%j-slurm.out

source ~/.bashrc
conda activate trl2

export HF_TOKEN= # Add token

python sft.py
python push.py

