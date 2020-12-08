#!/bin/bash

#SBATCH --job-name=combine_samples
#SBATCH --output=combine.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=150GB
#SBATCH --time=10:00:00
# #SBATCH --gres=gpu:1

module purge

singularity exec --nv \
            --overlay /scratch/sm8383/sbi-fermi-overlay.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            bash -c "source /ext3/env.sh; conda activate sbi-fermi; cd /scratch/sm8383/sbi-fermi/; \
            ./combine_samples.py --regex train 'train_\d+; --dir /scratch/sm8383/sbi-fermi/"