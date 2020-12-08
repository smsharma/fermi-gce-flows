#!/bin/bash

#SBATCH --job-name=simulate
#SBATCH --output=train.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=12:59:00
#SBATCH --gres=gpu:1

module purge

singularity exec --nv \
            --overlay /scratch/sm8383/sbi-fermi-overlay.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            bash -c "source /ext3/env.sh; conda activate sbi-fermi; cd /scratch/sm8383/sbi-fermi/; \
            python -u train.py --sample train --name vanilla"