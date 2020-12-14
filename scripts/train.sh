#!/bin/bash

#SBATCH --job-name=simulate
#SBATCH --output=train.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=12:59:00
#SBATCH --gres=gpu:1

conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi/
python -u train.py --sample train --name vanilla