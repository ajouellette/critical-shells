# critical-shells

Code for locating and analyzing critcal shells for dark matter halos in GADGET-4 simulations.

## Installation
Create a conda environment
```
$ conda create -n critical-shells python=3.9 boost numpy scipy numba h5py matplotlib mpi4py astropy
```

To run tests, also install packages listed in `requirements_dev.txt`.

Install in development mode:
```
$ conda activate critical-shells
$ cd /path/to/critical-shells
$ pip install -e .
```
