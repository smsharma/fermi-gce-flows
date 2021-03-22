#!/bin/bash

#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:59:00
#SBATCH --mem=32GB
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=sm8383@nyu.edu
#SBATCH --gres=gpu:1

source ~/.bashrc
port=$(shuf -i 6000-9999 -n 1)

/usr/bin/ssh -N -f -R $port:localhost:$port log-0
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :

If you are working in NYU campus, please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@greene.hpc.nyu.edu

If you are working off campus, you should already have ssh tunneling setup through HPC bastion host, 
that you can directly login to greene with command

ssh $USER@greene

Please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@greene

Step 2:

Keep the iTerm windows in the previouse step open. Now open browser, find the line with

The Jupyter Notebook is running at: $(hostname)

the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

you should be able to connect to jupyter notebook running remotly on prince compute node with above url

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

conda activate sbi-fermi
cd /scratch/sm8383/
jupyter lab --no-browser --port ${port} --notebook-dir=${pwd}


