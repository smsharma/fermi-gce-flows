#!/bin/bash

#SBATCH --job-name=combine_samples
#SBATCH --output=combine.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --time=10:00:00
# #SBATCH --gres=gpu:1

conda activate
cd /home/sm8383/sbi-fermi/

./combine_samples.py --regex train "train_\d+" --dir /scratch/sm8383/sbi-fermi/
