#!/bin/bash

#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=halosfind
#SBATCH --partition=caps
#SBATCH --output=slurm/logs/%x.out
#SBATCH --error=slurm/logs/%x.err
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=aaronjo2@illinois.edu

n=512  # select simulation

source ~/.bashrc
conda activate conda-cosmo
snap100=$(python slurm/get_snapnum.py ../dm-l256-n$n-a100 100)

date
srun python -u scripts/critical_shell.py $PROJ/dm-l256-n$n-a100 $snap100
