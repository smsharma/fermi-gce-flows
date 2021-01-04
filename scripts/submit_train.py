import os

batch = """#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --time=30:59:00
#SBATCH --gres=gpu:rtx8000

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi/
"""

batch_size_list = [64, 256]
fc_dims_list = [[[-1, 2048], [2048, 512], [512, 96]], 
                [[-1, 2048], [2048, 512]]]
maf_num_transforms_list = [2, 8]

for maf_num_transforms in maf_num_transforms_list:
    for batch_size in batch_size_list:
        for fc_dims in fc_dims_list:
            batchn = batch + "\n"
            batchn += "python -u train.py --sample train_float_all_ModelO --name gce_float_all_ModelO --maf_num_transforms {} --fc_dims '{}' --batch_size {}".format(maf_num_transforms, fc_dims, batch_size)
            fname = "batch/submit.batch"
            f = open(fname, "w")
            f.write(batchn)
            f.close()
            os.system("chmod +x " + fname)
            os.system("sbatch " + fname)
