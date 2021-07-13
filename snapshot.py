import h5py
import numpy as np

# Assumptions about GADGET hdf5 file:
#   single snapshot file contains all output data for given step
#   file name is of the format {}_number.hdf5
#   cosmological simulation (in comoving coordinates)
#   default code units: Mpc, km/s, 1e10 Msun
#   only dark matter particles - PartType1
#   all particles have the same mass

class Snapshot:

    def __init__(self, fname):
        self._hdf = h5py.File(fname)
        self.snap_num = int(fname.split('/')[-1].split('_')[-1].split('.')[0])

        self.a = self._hdf["Header"].attrs["Time"]
        self.z = self._hdf["Header"].attrs["Redshift"]
        self.box_size = self._hdf["Header"].attrs["BoxSize"]

        # Cosmology parameters
        self.OmegaLambda = self._hdf["Parameters"].attrs["OmegaLambda"]
        self.OmegaMatter = self._hdf["Parameters"].attrs["Omega0"]
        self.Hubble0 = self._hdf["Parameters"].attrs["Hubble"] * self._hdf["Parameters"].attrs["HubbleParam"]
        self.Hubble = self.Hubble0 * np.sqrt(self.OmegaMatter * self.a**-3 + (1-self.OmegaMatter-self.OmegaLambda) * self.a**-2 + self.OmegaLambda)


class ParticleData(Snapshot):

    def __init__(self, fname):
        super().__init__(fname)
        self.n_parts = self._hdf["Header"].attrs["NumPart_Total"][1]
        self.part_mass = self._hdf["Header"].attrs["MassTable"][1]

        self.pos = self._hdf["PartType1"]["Coordinates"][:]
        self.vel = self._hdf["PartType1"]["Velocities"][:]
        self.ids = self._hdf["PartType1"]["ParticleIDs"][:]

        self._hdf.close()

    def select_ids(self, ids):
        """Return indicies of particles given list of ids."""
        return np.where(np.in1d(self.ids, ids))


class HaloCatalog(Snapshot):

    def __init__(self, fname):
        super().__init__(fname)
        self.n_halos = self._hdf["Header"].attrs["Ngroups_Total"]

        self.pos = self._hdf["Group"]["GroupPos"][:]
        self.vel = self._hdf["Group"]["GroupVel"][:]
        self.offsets = self._hdf["Group"]["GroupOffsetType"][:,1]
        self.lengths = self._hdf["Group"]["GroupLen"][:]

        self._hdf.close()

    def get_particle_ids(self, halo_i, particle_data):
        """Return ids of particles that are part of the given halo."""
        offset = self.offsets[halo_i]
        length = self.lengths[halo_i]
        return particle_data.ids[offset:offset+length]

