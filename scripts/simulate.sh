#!/bin/bash

#SBATCH --job-name=simulate
#SBATCH --output=log_simulate_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
# #SBATCH --gres=gpu:1

conda activate
cd /scratch/sm8383/sbi-fermi/

python -u simulate.py -n 10000 --name train_${SLURM_ARRAY_TASK_ID} --dir /scratch/sm8383/sbi-fermi/
