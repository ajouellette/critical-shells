#!/bin/bash

#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=fullanalysis
#SBATCH --partition=caps
#SBATCH --output=slurm/logs/%x.out
#SBATCH --error=slurm/logs/%x.err
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=aaronjo2@illinois.edu

n=512

source ~/.bashrc
conda activate conda-cosmo
data=$PROJ/dm-l256-n$n-a100
snap100=$(python slurm/get_snapnum.py $PROJ/dm-l256-n$n-a100 100)
snap1=$(python slurm/get_snapnum.py $PROJ/dm-l256-n$n-a100 1)

date
srun -d singleton python -u scripts/critical_shell.py $data $snap100 && \
srun -d singleton python -u scripts/vr_crit_shell.py $data $snap100 && \
srun -d singleton python -u scripts/crit_shell_fof100.py $data $snap100 && \
srun -d singleton python -u scripts/crit_shell_fof.py $data $snap1 && \
srun -d singleton python -u scripts/fof_optim.py $data $snap1

mv slurm/logs/"$SLURM_JOB_NAME".out slurm/logs/"$SLURM_JOB_NAME"-"$n".out
mv slurm/logs/"$SLURM_JOB_NAME".err slurm/logs/"$SLURM_JOB_NAME"-"$n".err
