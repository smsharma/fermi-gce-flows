#!/bin/bash

#SBATCH --job-name=combine_samples
#SBATCH --output=combine.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=400GB
#SBATCH --time=00:59:00
# #SBATCH --gres=gpu:1

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi
./combine_samples.py --regex train_ModelF_gamma_fix_1M 'train_ModelF_gamma_fix_\d+' --dir /scratch/sm8383/sbi-fermi/
# ./combine_samples.py --regex train_ModelA_gamma_fix_1M 'train_ModelA_gamma_fix_\d+' --dir /scratch/sm8383/sbi-fermi/

