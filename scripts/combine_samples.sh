#!/bin/bash

#SBATCH --job-name=combine_samples
#SBATCH --output=combine.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=550GB
#SBATCH --time=01:30:00
# #SBATCH --gres=gpu:1

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi

./combine_samples.py --regex train_ModelO_gamma_fix_thin_disk_1p3M 'train_ModelO_gamma_fix_thin_disk_\d+' --dir /scratch/sm8383/sbi-fermi/

