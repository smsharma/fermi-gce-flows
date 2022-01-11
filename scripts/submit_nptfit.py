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

cd /scratch/sm8383/fermi-gce-flows
"""

# # Runs on data

# # Various configurations
# sample_list = ["fermi_data_thin_disk_1000", "fermi_data_ModelO_1000", "fermi_data_thin_disk_ModelA_1000", "fermi_data_thin_disk_ModelF_1000", "fermi_data_thin_disk_new_ps_priors_1000"]
# new_ps_priors_list = [0, 0, 0, 0, 1]
# disk_type_list = ["thin", "thick", "thin", "thin", "thin"]
# diffuse_list = ["ModelO", "ModelO", "ModelA", "ModelF", "ModelO"]

# for sample_name, new_ps_priors, disk_type, diffuse in zip(sample_list, new_ps_priors_list, disk_type_list, diffuse_list):
#     batchn = batch + "\n"
#     batchn += "python nptfit.py --sample_name {} --n_cpus 24 --r_outer 25 --n_live 1000 --disk_type {} --i_mc -1 --diffuse {} --new_ps_priors {}".format(sample_name, disk_type, diffuse, new_ps_priors)
#     fname = "batch/submit.batch"
#     f = open(fname, "w")
#     f.write(batchn)
#     f.close()
#     os.system("chmod +x " + fname)
#     os.system("sbatch " + fname)


# Tests on simulations

# Various configurations
sample_list = ["ModelO_DM_only_mismo"]
new_ps_priors_list = [0]
disk_type_list = ["thin"]
diffuse_list = ["ModelO"]

for sample_name, new_ps_priors, disk_type, diffuse in zip(sample_list, new_ps_priors_list, disk_type_list, diffuse_list):
    for i_mc in range(5):
        batchn = batch + "\n"
        batchn += "python nptfit.py --sample_name {} --n_cpus 24 --r_outer 25 --n_live 1000 --disk_type {} --i_mc {} --diffuse {} --new_ps_priors {}".format(sample_name,  disk_type, i_mc, diffuse, new_ps_priors)
        fname = "batch/submit.batch"
        f = open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname)
        os.system("sbatch " + fname)