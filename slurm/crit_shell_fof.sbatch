#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=fof1
#SBATCH --partition=caps
#SBATCH --output=slurm/logs/%x.out
#SBATCH --error=slurm/logs/%x.err
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=aaronjo2@illinois.edu

n=512

source ~/.bashrc
conda activate conda-cosmo
snap1=$(python slurm/get_snapnum.py $PROJ/dm-l256-n$n-a100 1)

date
srun python -u scripts/crit_shell_fof.py $PROJ/dm-l256-n$n-a100 $snap1
