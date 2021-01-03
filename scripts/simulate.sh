#!/bin/bash

#SBATCH --job-name=simulate
#SBATCH --output=simulate_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12GB
#SBATCH --time=02:59:00
# #SBATCH --gres=gpu:1

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi/
python -u simulate.py -n 1000 --name train_float_all_${SLURM_ARRAY_TASK_ID} --dir /scratch/sm8383/sbi-fermi/

