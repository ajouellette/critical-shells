#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=fof100
#SBATCH --partition=caps
#SBATCH --output=slurm/logs/%x.out
#SBATCH --error=slurm/logs/%x.err
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=aaronjo2@illinois.edu

set -u
set -e

n=512

source ~/.bashrc
conda activate conda-cosmo
snap100=$(python slurm/get_snapnum.py $PROJ/dm-l256-n$n-a100 100)

date
srun python -u scripts/crit_shell_fof100.py $PROJ/dm-l256-n$n-a100 $snap100
date

mv slurm/logs/"$SLURM_JOB_NAME".out slurm/logs/"$SLURM_JOB_NAME"-"$n".out
mv slurm/logs/"$SLURM_JOB_NAME".err slurm/logs/"$SLURM_JOB_NAME"-"$n".err
