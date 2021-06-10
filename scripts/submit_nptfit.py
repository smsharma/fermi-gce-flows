import sys, os
import numpy as np

batch = """#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=24
#SBATCH -t 47:59:00
#SBATCH --mem=24GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=sm8383@nyu.edu

module load gsl/intel/2.6 
conda activate sbi-fermi

cd /scratch/sm8383/sbi-fermi
"""

sample_list = ["fermi_data"]
n_mc = 10

for sample_name in sample_list:
    for i_mc in [-1]:
        batchn = batch + "\n"
        batchn += "python nptfit.py --sample_name {} --n_cpus 24 --r_outer 25 --n_live 1000 --i_mc {}".format(sample_name, i_mc)
        fname = "batch/submit.batch"
        f = open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname)
        os.system("sbatch " + fname)


# sample_list = ["ModelA_PS_only_mismo", "ModelA_DM_only_mismo"]
# n_mc = 10

# for sample_name in sample_list:
#     for i_mc in range(n_mc):
#         batchn = batch + "\n"
#         batchn += "python nptfit.py --sample_name {} --n_cpus 24 --r_outer 25 --n_live 200 --i_mc {}".format(sample_name, i_mc)
#         fname = "batch/submit.batch"
#         f = open(fname, "w")
#         f.write(batchn)
#         f.close()
#         os.system("chmod +x " + fname)
#         os.system("sbatch " + fname)
