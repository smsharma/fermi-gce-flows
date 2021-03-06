import sys, os
import numpy as np

batch = """#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=36
#SBATCH -t 23:59:00
#SBATCH --mem=24GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=sm8383@nyu.edu

module load gsl/intel/2.6 
conda activate sbi-fermi

cd /scratch/sm8383/sbi-fermi
"""

diffuse_list = ["ModelO", "p6"]
ps_mask_list = ["0.8"]
transform_prior_on_s_list = [1]

for ps_mask_type in ps_mask_list:
    for diffuse in diffuse_list:
        for transform_prior_on_s in transform_prior_on_s_list:
            batchn = batch + "\n"
            sample_name = "runs_25_{}_{}_{}".format(diffuse, transform_prior_on_s, ps_mask_type)
            batchn += "python nptfit.py --diffuse {} --transform_prior_on_s {} --ps_mask_type {} --sample_name {} --n_cpus 36 --r_outer 25 --ps_mask_type default --n_live 400".format(diffuse, transform_prior_on_s, ps_mask_type, sample_name)
            fname = "batch/submit.batch"
            f = open(fname, "w")
            f.write(batchn)
            f.close()
            os.system("chmod +x " + fname)
            os.system("sbatch " + fname)
