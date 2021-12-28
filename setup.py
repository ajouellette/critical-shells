from setuptools import setup


PACKAGENAME = "critical-shells"
VERSION = "0.1.0"

if __name__ == "__main__":
    setup(
        name=PACKAGENAME,
        version=VERSION,
        author="Aaron Ouellette",
        author_email="aaronjo2@illinois.edu",
        description="",
        long_description="",
        install_requires=["numpy>=1.20",
                          "scipy>=1.7",
                          "sklearn>=1.0",
                          "numba>=0.54",
                          "h5py>=3.4",
                          "mpi4py>=3.1",
                          "astropy>=4.3",
                          "pyfof>=0.2"
                          ],
        packages=["gadgetutils"],
        url="https://github.com/ajouellette/critical-shells",
    )
