import os

batch = """#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=46GB
#SBATCH --time=36:59:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=sm8383@nyu.edu

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi/
"""

batch_size_list = [64, 128]
fc_dims_list = [[[-1, 2048], [2048, 128]],
                [[-1, 2048], [2048, 96]]]

maf_num_transforms_list = [2, 4]
maf_hidden_features_list = [128]
methods = ["snpe"]
activations = ["relu"]
kernel_size_list = [4]

for maf_num_transforms in maf_num_transforms_list:
    for maf_hidden_features in maf_hidden_features_list:
        for batch_size in batch_size_list:
            for fc_dims in fc_dims_list:
                for method in methods:
                    for activation in activations:
                        for kernel_size in kernel_size_list:
                            batchn = batch + "\n"
                            batchn += "python -u train.py --sample train_ModelO_gamma_fix --name gce_ModelO_gamma_fix_100k_v3 --method {} --maf_num_transforms {} --maf_hidden_features {} --fc_dims '{}' --batch_size {} --activation {} --kernel_size {}".format(method, maf_num_transforms, maf_hidden_features, fc_dims, batch_size, activation, kernel_size)
                            fname = "batch/submit.batch"
                            f = open(fname, "w")
                            f.write(batchn)
                            f.close()
                            os.system("chmod +x " + fname)
                            os.system("sbatch " + fname)