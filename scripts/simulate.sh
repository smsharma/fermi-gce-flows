#!/bin/bash

#SBATCH --job-name=simulate
#SBATCH --output=simulate_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6GB
#SBATCH --time=10:59:00
# #SBATCH --gres=gpu:1

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/fermi-gce-flows/

# python -u simulate.py -n 1000 --name train_ModelO_gamma_fix_thin_disk_rescale_${SLURM_ARRAY_TASK_ID} --dir /scratch/sm8383/fermi-gce-flows/ --gamma fix --dif ModelO --disk_type thin --new_ps_priors 0

python -u simulate.py -n 1000 --name train_ModelO_gamma_fix_thin_disk_negative_dm_prio_rescale_${SLURM_ARRAY_TASK_ID} --dir /scratch/sm8383/fermi-gce-flows/ --gamma fix --dif ModelO --disk_type thin --new_ps_priors 0 --prior_dm_negative 1