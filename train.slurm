#!/bin/bash
#SBATCH --job-name=train_a3
#SBATCH --ntasks=1
#SBATCH --account=rnt@gpu
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=out_train.out
#SBATCH --error=out_train.out

echo "Running job on $hostname"
cd /gpfswork/rech/rnt/uuj49ar/BRAMI_Victoria_a3

# Load conda enc
module load python/3.7.3
conda activate deepspeech_model2

set -x

python main.py --epochs 100 --lr 0.005 --momentum 0.9 --model_name vit-B_16 --in_size 224 --optimizer sgd --weight_decay 0.00 --scheduler exponential --aug_type flip crop center_crop
