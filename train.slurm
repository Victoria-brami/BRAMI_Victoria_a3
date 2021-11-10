#!/bin/bash
#SBATCH --job-name=train_a3
#SBATCH --ntasks=1
#SBATCH --account=rnt@gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --output=out_train.out
#SBATCH --error=out_train.out

echo "Running job on $hostname"
cd /gpfswork/rech/rnt/uuj49ar/BRAMI_Victoria_a3

# Load conda enc
module load python/3.7.3
conda activate deepspeech_model2

set -x

python main.py --epochs 50 --lr 0.001 --model_name resnet152 --optimizer adam --aug_type flip rotate colors erasing