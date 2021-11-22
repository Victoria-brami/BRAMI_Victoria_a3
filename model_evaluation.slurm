#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --ntasks=1
#SBATCH --account=rnt@gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=out_eval.out
#SBATCH --error=out_eval.out


echo "Running job on $hostname"
cd /gpfswork/rech/rnt/uuj49ar/BRAMI_Victoria_a3


module load python/3.7.3
conda activate deepspeech_model2


set -x

python3 evaluate.py --data ../cropped_bird_dataset --model_name vit-B_16-384 --outfile vit-B_16-384_yolo_cosine_scheduler_B20_model_34_kaggle.csv --model best_chkpts/vit-B_16_sgd_384_model_34.pth
