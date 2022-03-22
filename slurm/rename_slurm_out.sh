#!/bin/bash

# renames files of the form "job name"-"job id".* to "job name"-"sim name".*
# if target name already exists, first copy it to "job name"-"sim name"-old.*

# Usage ./rename_slurm_out.sh "job name" "job id" "sim name"
# renames files in current directory

set -e
set -u

if [ $# != 3 ]; then
    echo "Usage: $0 'job name' 'job id' 'sim dir'"
    exit 1
fi

job_name="$1"
job_id="$2"
sim_name=$(basename "$3")

# backup old output
if [ -f "$job_name"-"$sim_name".out ]; then
    echo Copying "$job_name"-"$sim_name".* to "$job_name"-"$sim_name"-old.*
    rename --verbose "$sim_name" "$sim_name"-old "$job_name"-"$sim_name".*
fi

echo Renaming "$job_name"-"$job_id".* to "$job_name"-"$sim_name".*
rename "$job_id" "$sim_name" "$job_name"-"$job_id".*
