#!/bin/bash

#SBATCH --job-name=simulate
#SBATCH --output=log_simulate_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=02:59:00
# #SBATCH --gres=gpu:1

module purge

singularity exec --nv \
            --overlay /scratch/sm8383/sbi-fermi-overlay.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            bash -c "source /ext3/env.sh; conda activate sbi-fermi; cd /scratch/sm8383/sbi-fermi/; \
            python -u simulate.py -n 1000 --name train_${SLURM_ARRAY_TASK_ID} --dir /scratch/sm8383/sbi-fermi/"