#!/bin/bash

#SBATCH --job-name=combine_samples
#SBATCH --output=combine.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=400GB
#SBATCH --time=01:30:00
# #SBATCH --gres=gpu:1

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/fermi-gce-flows

# ./combine_samples.py --regex train_ModelO_gamma_fix_thin_disk_rescale_1M 'train_ModelO_gamma_fix_thin_disk_rescale_\d+' --dir /scratch/sm8383/sbi-fermi/

./combine_samples.py --regex train_ModelO_gamma_fix_thin_disk_negative_dm_prio_rescale_1M 'train_ModelO_gamma_fix_thin_disk_negative_dm_prio_rescale_\d+' --dir /scratch/sm8383/fermi-gce-flows/

