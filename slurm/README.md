# SLURM batch scripts

These scripts are designed to submit jobs to the Illinois Campus Cluster. They
will run the analysis scripts and write output to `slurm/logs`.

The scripts assume that `$PROJ` gives the location of the project directory.
The simulation data is located at `$PROJ/<simulation-name>/`. This directory
contains all of the output snapshots, the FoF group catalogs, and the `outputs.txt`
file from the simulation setup.

`get_snapnum.py` uses `outputs.txt` to get the snapshot number that corresponds
to a given scale factor (usually 100 or 1).

The analysis scripts if given a simulation at the location `$PROJ/<simulation-name>/`
will usually write any output catalogs or analysis results to the location
`$PROJ/<simulation-name>-analysis`.
