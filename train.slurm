#!/bin/bash
#SBATCH --job-name=train_a3
#SBATCH --partition=gpu_p2
#SBATCH --ntasks=1
#SBATCH --account=rnt@gpu
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH --output=out_train.out
#SBATCH --error=out_train.out

echo "Running job on $hostname"
cd /gpfswork/rech/rnt/uuj49ar/BRAMI_Victoria_a3

# Load conda enc
module load python/3.7.3
conda activate deepspeech_model2

set -x

python main.py --data ../cropped_bird_dataset --epochs 100 --lr 0.001 --batch-size 20 --momentum 0.9 --model_name vit-B_16 --in_size 384 --optimizer sgd --weight_decay 0.000 --scheduler cosine --aug_type flip random_crop crop colors
