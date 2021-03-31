import os
import numpy as np

batch = """#!/bin/bash

#SBATCH --job-name=summarize
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=12:59:00

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi/
"""

range_list = np.arange(0, 500, 20)
file_range_list = np.transpose([range_list[:-1],range_list[1:]])

for file_range in [[0, 500]]:
    batchn = batch + "\n"
    batchn += "python -u summarize.py --sample train_ModelO_gamma_fix_500k --n_files '{}' --do_pca".format(list(file_range))
    fname = "batch/submit.batch"
    f = open(fname, "w")
    f.write(batchn)
    f.close()
    os.system("chmod +x " + fname)
    os.system("sbatch " + fname)

for file_range in file_range_list:
    batchn = batch + "\n"
    batchn += "python -u summarize.py --sample train_ModelO_gamma_fix_500k --n_files '{}' --do_power_spectrum".format(list(file_range))
    fname = "batch/submit.batch"
    f = open(fname, "w")
    f.write(batchn)
    f.close()
    os.system("chmod +x " + fname)
    os.system("sbatch " + fname)