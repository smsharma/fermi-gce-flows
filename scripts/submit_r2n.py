import os
import numpy as np

batch = """#!/bin/bash

#SBATCH --job-name=r2n
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=2:59:00

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi/
"""

n_files = 50
i_starts = np.arange(0, 1200, n_files)

for i_start in i_starts:
    batchn = batch + "\n"
    batchn += "python -u ring2nest.py --i_start {} --n_files {}".format(i_start, n_files)
    fname = "batch/submit.batch"
    f = open(fname, "w")
    f.write(batchn)
    f.close()
    os.system("chmod +x " + fname)
    os.system("sbatch " + fname)