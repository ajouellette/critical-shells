#!/bin/bash

#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=energies_sh
#SBATCH --partition=caps
#SBATCH --output=slurm/logs/%x.out
#SBATCH --error=slurm/logs/%x.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aaronjo2@illinois.edu

set -e

source ~/.bashrc
conda activate conda-cosmo

date
srun python -u scripts/energy.py $PROJ/dm-l256-n256-a100 subhalos
date
