#!/bin/bash

#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=fullanalysis
#SBATCH --partition=caps
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=aaronjo2@illinois.edu

set -e

data_dir="$1"
if [ ! -d "$data_dir" ]; then
    echo Could not find data directory "$data_dir"
    exit 1
fi

source ~/.bashrc
conda activate conda-cosmo
snap100=$(python slurm/get_snapnum.py $data_dir 100)
snap1=$(python slurm/get_snapnum.py $data_dir 1)

date
srun -d singleton python -u scripts/critical_shell.py $data_dir $snap100 && \
srun -d singleton python -u scripts/vr_crit_shell.py $data_dir $snap100 && \
srun -d singleton python -u scripts/crit_shell_fof100.py $data_dir $snap100 && \
srun -d singleton python -u scripts/crit_shell_fof.py $data_dir $snap1 && \
srun -d singleton python -u scripts/fof_optim.py $data_dir $snap1
date

cd slurm/logs
../rename_slurm_out.sh "$SLURM_JOB_NAME" "$SLURM_JOB_ID" "$data_dir"
