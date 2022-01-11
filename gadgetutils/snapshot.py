import h5py
import numpy as np
from . import utils

# Assumptions about GADGET hdf5 file:
#   single snapshot file contains all output data for given step
#   file name is of the format {}_number.hdf5
#   cosmological simulation (in comoving coordinates)
#   default code units: h^-1 Mpc, km/s, 1e10 h^-1 Msun
#   only dark matter particles - PartType1
#   all particles have the same mass

# Note: all data loaded directly into memory
#   might be a problem with very large simulations
# For a 256**3 particle sim in single precision a ParticleData object will use ~0.44GB
#  a 512**3 particle sim will use ~3.5GB


class Snapshot:

    def __init__(self, fname):
        self.file_name = fname
        self.snap_num = int(fname.split('/')[-1].split('_')[-1].split('.')[0])
        with h5py.File(fname) as f:
            self.a = f["Header"].attrs["Time"]
            self.z = f["Header"].attrs["Redshift"]
            self.box_size = f["Header"].attrs["BoxSize"]

            # Cosmology parameters
            self.OmegaLambda = f["Parameters"].attrs["OmegaLambda"]
            self.OmegaMatter = f["Parameters"].attrs["Omega0"]
            self.Hubble0 = f["Parameters"].attrs["Hubble"] \
                * f["Parameters"].attrs["HubbleParam"]
            self.h = self.Hubble0 / 100
            self.Hubble = self.Hubble0 * np.sqrt(self.OmegaMatter * self.a**-3
                        + (1-self.OmegaMatter-self.OmegaLambda) * self.a**-2
                        + self.OmegaLambda)


class ParticleData(Snapshot):
    """Load GADGET-4 snapshots to access particle data.
    """

    def __init__(self, fname, load_vels=True, load_ids=True):
        super().__init__(fname)
        with h5py.File(fname) as f:
            self.n_parts = f["Header"].attrs["NumPart_Total"][1]
            self.part_mass = f["Header"].attrs["MassTable"][1] * 1e10
            self.mean_particle_sep = self.box_size / self.n_parts**(1/3)
            self.mean_matter_density = self.n_parts * self.part_mass / self.box_size**3
            self.flat_crit_density = self.mean_matter_density / self.OmegaMatter

            self.pos = f["PartType1"]["Coordinates"][:]
            self.vel = None
            self.ids = None
            if load_vels:
                self.vel = f["PartType1"]["Velocities"][:]
            if load_ids:
                self.ids = f["PartType1"]["ParticleIDs"][:]

    def select_ids(self, ids):
        """Return indicies of particles given list of ids."""
        if self.ids is None:
            raise RuntimeError("Cannot select particles, snapshot loaded with load_ids=False.")
        return np.nonzero(np.isin(self.ids, ids))


class HaloCatalog(Snapshot):
    """Load GADGET-4 snapshot halo data (FoF and subfind clusters).
    """

    def __init__(self, fname):
        super().__init__(fname)

        with h5py.File(fname) as f:
            self.n_halos = f["Header"].attrs["Ngroups_Total"]

            self.pos = f["Group"]["GroupPos"][:]
            self.vel = f["Group"]["GroupVel"][:]
            self.masses = f["Group"]["GroupMass"][:] * 1e10

            self.offsets = f["Group"]["GroupOffsetType"][:,1]
            self.lengths = f["Group"]["GroupLen"][:]

    def get_particle_ids(self, halo_i, particle_data):
        """Return ids of particles that are part of the given halo."""
        offset = self.offsets[halo_i]
        length = self.lengths[halo_i]
        return particle_data.ids[offset:offset+length]

    def calc_hmf(self, bins):
        """Calculate HMF of FoF clusters."""
        return utils.calc_hmf(bins, self.masses, self.box_size)
