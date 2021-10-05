import sys, os
import numpy as np

batch = """#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=24
#SBATCH -t 47:59:00
#SBATCH --mem=24GB
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=sm8383@nyu.edu

module load gsl/intel/2.6 
conda activate sbi-fermi

cd /scratch/sm8383/sbi-fermi
"""

sample_list = ["fermi_data_thin_disk_new_ps_priors_1000"]

for sample_name in sample_list:
    for i_mc in [-1]:
        batchn = batch + "\n"
        batchn += "python nptfit.py --sample_name {} --n_cpus 24 --r_outer 25 --n_live 1000 --disk_type thin --i_mc {} --diffuse ModelO --new_ps_priors 1".format(sample_name, i_mc)
        fname = "batch/submit.batch"
        f = open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname)
        os.system("sbatch " + fname)

# sample_list = ["fermi_data_ModelA_1000"]

# for sample_name in sample_list:
#     for i_mc in [-1]:
#         batchn = batch + "\n"
#         batchn += "python nptfit.py --sample_name {} --n_cpus 24 --r_outer 25 --n_live 1000 --disk_type thick --i_mc {} --diffuse ModelA".format(sample_name, i_mc)
#         fname = "batch/submit.batch"
#         f = open(fname, "w")
#         f.write(batchn)
#         f.close()
#         os.system("chmod +x " + fname)
#         os.system("sbatch " + fname)

# sample_list = ["fermi_data_ModelF_1000"]

# for sample_name in sample_list:
#     for i_mc in [-1]:
#         batchn = batch + "\n"
#         batchn += "python nptfit.py --sample_name {} --n_cpus 24 --r_outer 25 --n_live 1000 --disk_type thick --i_mc {} --diffuse ModelF".format(sample_name, i_mc)
#         fname = "batch/submit.batch"
#         f = open(fname, "w")
#         f.write(batchn)
#         f.close()
#         os.system("chmod +x " + fname)
#         os.system("sbatch " + fname)