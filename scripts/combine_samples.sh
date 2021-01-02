  
#!/bin/bash

#SBATCH --job-name=combine_samples
#SBATCH --output=combine.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200GB
#SBATCH --time=01:59:00
# #SBATCH --gres=gpu:1

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi
./combine_samples.py --regex train_float_all 'train_float_all_\d+' --dir /scratch/sm8383/sbi-fermi/

