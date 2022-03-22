#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=vr_analysis
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

date
srun python -u scripts/vr_crit_shell.py $data_dir $snap100
date

../rename_slurm_out.sh "$SLURM_JOB_NAME" "$SLURM_JOB_ID" "$data_dir"
