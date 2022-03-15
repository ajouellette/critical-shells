from setuptools import setup
from Cython.Build import cythonize


PACKAGENAME = "critical-shells"
VERSION = "0.1.0"

if __name__ == "__main__":
    setup(
        name=PACKAGENAME,
        version=VERSION,
        author="Aaron Ouellette",
        author_email="aaronjo2@illinois.edu",
        license="LICENSE",
        description="GADGET simulation analysis",
        long_description=open("README.md").read(),
        setup_requires=["Cython>=0.29",
            ],
        install_requires=["numpy>=1.20",
                          "scipy>=1.7",
                          "scikit-learn>=1.0",
                          "numba>=0.54",
                          "h5py>=3.4",
                          "mpi4py>=3.1",
                          "astropy>=4.3",
                          "pyfof @ git+https://github.com/ajouellette/pyfof"
                          ],
        packages=["gadgetutils"],
        ext_modules = cythonize("gadgetutils/potential.pyx", language_level=3),
        scripts=["scripts/parse_gadget_out.py"],
        url="https://github.com/ajouellette/critical-shells",
    )
