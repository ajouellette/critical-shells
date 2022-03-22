#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=fof1
#SBATCH --partition=caps
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=aaronjo2@illinois.edu

set -e

data_dir="$1"
if [ -d "$data_dir" ]; then
    echo Could not find data directory "$data_dir"
    exit 1
fi

source ~/.bashrc
conda activate conda-cosmo
snap1=$(python slurm/get_snapnum.py $data_dir 1)

date
srun python -u scripts/crit_shell_fof.py $data_dir $snap1
date

# rename outputs to something recongnizable
../rename_slurm_out.sh "$SLURM_JOB_NAME" "$SLURM_JOB_ID" "$data_dir"
