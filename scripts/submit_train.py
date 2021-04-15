import os

batch = """#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=42GB
#SBATCH --time=39:59:00
#SBATCH --gres=gpu:1
##SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=siddharthmishra19@gmail.com

source ~/.bashrc
conda activate sbi-fermi
cd /scratch/sm8383/sbi-fermi/
"""

##########################
# Explore configurations #
##########################

batch_size_list = [128]
fc_dims_list = [[[-1, 2048], [2048, 128]], [[-1, 2048], [2048, 512]], [[-1, 2048], [2048, 512], [512, 128]]]
maf_num_transforms_list = [4]
maf_hidden_features_list = [128]
methods = ["snpe"]
activations = ["relu"]
kernel_size_list = [4]
n_neighbours_list = [8, 20]
conv_channel_configs = ["standard"]
laplacian_types = ["normalized"]
conv_types = ["chebconv"]
aux_summaries = ["None"]
n_aux_list = [2]

for n_neighbours in n_neighbours_list:
    for maf_num_transforms in maf_num_transforms_list:
        for maf_hidden_features in maf_hidden_features_list:
            for batch_size in batch_size_list:
                for fc_dims in fc_dims_list:
                    for method in methods:
                        for activation in activations:
                            for kernel_size in kernel_size_list:
                                for laplacian_type in laplacian_types:
                                    for conv_type in conv_types:
                                        for conv_channel_config in conv_channel_configs:
                                            for aux_summary, n_aux in zip(aux_summaries, n_aux_list):
                                                batchn = batch + "\n"
                                                batchn += "python -u train.py --sample train_ModelO_gamma_default_1M --name gce_ModelO_gamma_default_1M --method {} --maf_num_transforms {} --maf_hidden_features {} --fc_dims '{}' --batch_size {} --activation {} --kernel_size {} --laplacian_type {} --conv_type {} --conv_channel_config {} --aux_summary {} --n_aux {} --n_neighbours {}".format(method, maf_num_transforms, maf_hidden_features, fc_dims, batch_size, activation, kernel_size, laplacian_type, conv_type, conv_channel_config, aux_summary, n_aux, n_neighbours)
                                                fname = "batch/submit.batch"
                                                f = open(fname, "w")
                                                f.write(batchn)
                                                f.close()
                                                os.system("chmod +x " + fname)
                                                os.system("sbatch " + fname)

# ##################
# # Just summaries #
# ##################

# batch_size_list = [128]
# maf_num_transforms_list = [4, 12]
# maf_hidden_features_list = [128, 512]
# methods = ["snpe"]
# activations = ["relu"]
# summaries = ["pca_96", "pspec_4"]

# for maf_num_transforms in maf_num_transforms_list:
#     for maf_hidden_features in maf_hidden_features_list:
#         for batch_size in batch_size_list:
#             for method in methods:
#                 for activation in activations:
#                     for summary in summaries:
#                         batchn = batch + "\n"
#                         batchn += "python -u train.py --sample train_ModelO_gamma_fix_480k --name gce_ModelO_gamma_fix_480k --method {} --maf_num_transforms {} --maf_hidden_features {} --batch_size {} --activation {} --summary {}".format(method, maf_num_transforms, maf_hidden_features, batch_size, activation, summary)
#                         fname = "batch/submit.batch"
#                         f = open(fname, "w")
#                         f.write(batchn)
#                         f.close()
#                         os.system("chmod +x " + fname)
#                         os.system("sbatch " + fname)