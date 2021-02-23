#!/bin/bash

#SBATCH --job-name=summarize
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6GB
#SBATCH --time=12:59:00

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi/
python -u summarize.py --sample train_ModelO_gamma_fix --n_files 500 --do_histogram --do_power_spectrum

