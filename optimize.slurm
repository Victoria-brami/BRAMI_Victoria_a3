#!/bin/bash
#SBATCH --job-name=optimize
#SBATCH --ntasks=1
#SBATCH --account=rnt@gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --output=optuna_out.out
#SBATCH --error=optuna_out.out

echo "Running job on $hostname"
cd /gpfswork/rech/rnt/uuj49ar/BRAMI_Victoria_a3

# Load conda enc
module load python/3.7.3
conda activate deepspeech_model2

set -x

python optuna_training.py
