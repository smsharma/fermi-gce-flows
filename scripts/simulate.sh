#!/bin/bash

#SBATCH --job-name=simulate
#SBATCH --output=simulate_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=05:59:00
# #SBATCH --gres=gpu:1

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi/
# python -u simulate.py -n 1000 --name train_ModelO_gamma_default_${SLURM_ARRAY_TASK_ID} --dir /scratch/sm8383/sbi-fermi/ --gamma default
python -u simulate.py -n 1000 --name train_ModelO_gamma_fix_${SLURM_ARRAY_TASK_ID} --dir /scratch/sm8383/sbi-fermi/ --gamma fix