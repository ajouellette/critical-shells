#!/bin/bash

#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=energies_sh
#SBATCH --partition=caps
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aaronjo2@illinois.edu

set -e

data_dir="$1"
if [ ! -d "$data_dir" ]; then
    echo Could not find data directory "$data_dir"
    exit 1
fi

source ~/.bashrc
conda activate conda-cosmo

date
srun python -u scripts/energy.py $data_dir subhalos
date

cd slurm/logs
../rename_slurm_out.sh "$SLURM_JOB_NAME" "$SLURM_JOB_ID" "$data_dir"
