#!/bin/bash

#SBATCH --job-name=simulate
#SBATCH --output=simulate_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6GB
#SBATCH --time=05:59:00
# #SBATCH --gres=gpu:1

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi/

# python -u simulate.py -n 1000 --name train_ModelO_gamma_fix_thin_disk_${SLURM_ARRAY_TASK_ID} --dir /scratch/sm8383/sbi-fermi/ --gamma fix --dif ModelO --disk_type thin

# python -u simulate.py -n 1000 --name train_ModelF_gamma_fix_thin_disk_${SLURM_ARRAY_TASK_ID} --dir /scratch/sm8383/sbi-fermi/ --gamma fix --dif ModelF --disk_type thin

python -u simulate.py -n 1000 --name train_ModelA_gamma_fix_thin_disk_${SLURM_ARRAY_TASK_ID} --dir /scratch/sm8383/sbi-fermi/ --gamma fix --dif ModelA --disk_type thin