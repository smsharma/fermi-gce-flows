#!/bin/bash

#SBATCH --job-name=combine_samples
#SBATCH --output=combine.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=450GB
#SBATCH --time=02:59:00
# #SBATCH --gres=gpu:1

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi
./combine_samples.py --regex train_float_all_ModelO 'train_float_all_ModelO_\d+' --dir /scratch/sm8383/sbi-fermi/

