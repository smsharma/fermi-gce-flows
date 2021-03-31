import os

batch = """#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=42GB
#SBATCH --time=38:59:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=siddharthmishra19@gmail.com

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi/
"""

batch_size_list = [128]
fc_dims_list = [[[-1, 2048], [2048, 512], [512, 256]],
                [[-1, 2048], [2048, 128]]]
maf_num_transforms_list = [8]
maf_hidden_features_list = [128]
methods = ["snpe"]
activations = ["relu", "selu"]
kernel_size_list = [4]
laplacian_types = ["combinatorial"]
chebconvs = ["deepsphere"]

for maf_num_transforms in maf_num_transforms_list:
    for maf_hidden_features in maf_hidden_features_list:
        for batch_size in batch_size_list:
            for fc_dims in fc_dims_list:
                for method in methods:
                    for activation in activations:
                        for kernel_size in kernel_size_list:
                            for laplacian_type in laplacian_types:
                                for chebconv in chebconvs:
                                    batchn = batch + "\n"
                                    batchn += "python -u train.py --sample train_ModelO_gamma_fix_500k --name gce_ModelO_gamma_fix_500k --method {} --maf_num_transforms {} --maf_hidden_features {} --fc_dims '{}' --batch_size {} --activation {} --kernel_size {} --laplacian_type {} --chebconv {}".format(method, maf_num_transforms, maf_hidden_features, fc_dims, batch_size, activation, kernel_size, laplacian_type, chebconvs)
                                    fname = "batch/submit.batch"
                                    f = open(fname, "w")
                                    f.write(batchn)
                                    f.close()
                                    os.system("chmod +x " + fname)
                                    os.system("sbatch " + fname)