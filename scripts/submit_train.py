import os

batch = """#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=36:59:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=sm8383@nyu.edu

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi/
"""

# batch_size_list = [256]
# maf_num_transforms_list = [4]
# summaries_list = ["hist_96", "pspec_4"]

# for maf_num_transforms in maf_num_transforms_list:
#     for batch_size in batch_size_list:
#         for summary in summaries_list:
#             batchn = batch + "\n"
#             batchn += "python -u train.py --sample train_ModelO_gamma_fix --name gce_ModelO_gamma_fix --maf_num_transforms {} --batch_size {} --summary {}".format(maf_num_transforms, batch_size, summary)
#             fname = "batch/submit.batch"
#             f = open(fname, "w")
#             f.write(batchn)
#             f.close()
#             os.system("chmod +x " + fname)
#             os.system("sbatch " + fname)

# batch_size_list = [64, 256]
# fc_dims_list = [[[-1, 2048], [2048, 512], [512, 96]], 
#                 [[-1, 2048], [2048, 512]]]
# maf_num_transforms_list = [4]
# methods = ["snpe", "snre"]

# batch_size_list = [256, 1024]
# fc_dims_list = [[[-1, 2048], [2048, 512], [512, 128]], 
#                 [[-1, 2048], [2048, 128]]]

# maf_num_transforms_list = [4, 12]
# methods = ["snpe"]
# activations = ["relu", "selu"]

# for maf_num_transforms in maf_num_transforms_list:
#     for batch_size in batch_size_list:
#         for fc_dims in fc_dims_list:
#             for method in methods:
#                 for activation in activations:
#                     batchn = batch + "\n"
#                     batchn += "python -u train.py --sample train_ModelO_gamma_fix --name gce_ModelO_gamma_fix_100k --method {} --maf_num_transforms {} --fc_dims '{}' --batch_size {} --activation {}".format(method, maf_num_transforms, fc_dims, batch_size, activation)
#                     fname = "batch/submit.batch"
#                     f = open(fname, "w")
#                     f.write(batchn)
#                     f.close()
#                     os.system("chmod +x " + fname)
#                     os.system("sbatch " + fname)

batch_size_list = [256]
fc_dims_list = [[[-1, 2048], [2048, 64]], 
                [[-1, 2048], [2048, 96]]]

maf_num_transforms_list = [4]
methods = ["snpe"]
activations = ["relu"]

for maf_num_transforms in maf_num_transforms_list:
    for batch_size in batch_size_list:
        for fc_dims in fc_dims_list:
            for method in methods:
                for activation in activations:
                    batchn = batch + "\n"
                    batchn += "python -u train.py --sample train_ModelO_gamma_fix --name gce_ModelO_gamma_fix_100k --method {} --maf_num_transforms {} --fc_dims '{}' --batch_size {} --activation {}".format(method, maf_num_transforms, fc_dims, batch_size, activation)
                    fname = "batch/submit.batch"
                    f = open(fname, "w")
                    f.write(batchn)
                    f.close()
                    os.system("chmod +x " + fname)
                    os.system("sbatch " + fname)