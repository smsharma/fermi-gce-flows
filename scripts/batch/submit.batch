#!/bin/bash
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

python nptfit.py --sample_name ModelO_DM_only --n_cpus 24 --r_outer 25 --n_live 1000 --disk_type thin --i_mc 4 --diffuse ModelO --new_ps_priors 0