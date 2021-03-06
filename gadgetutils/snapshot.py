import warnings
import h5py
import numpy as np
from scipy.spatial import KDTree
from . import utils


def load(fname, **kwargs):
    """Load a .hdf5 GADGET output file."""
    particle_snap = False
    group_snap = False
    with h5py.File(fname) as f:
        try:
            f["Header"].attrs["Ngroups_Total"]
            group_snap = True
        except KeyError:
            try:
                f["PartType1"]
                particle_snap = True
            except KeyError:
                pass
    if group_snap:
        return HaloCatalog(fname, **kwargs)
    elif particle_snap:
        return ParticleData(fname, **kwargs)
    else:
        raise ValueError(f"File {fname} not recognized as a particle or group snapshot")


class Snapshot:
    """Base class to load GADGET-4 particle or group snapshot files.

    Assumptions about GADGET hdf5 file:
        single snapshot file contains all output data for given step
        file name is of the format {}_number.hdf5
        cosmological simulation (in comoving coordinates)
        default code units: h^-1 Mpc, km/s, 1e10 h^-1 Msun
    """

    def __init__(self, fname):
        self.file_name = fname
        try:
            self.snap_num = int(fname.split('/')[-1].split('_')[-1].split('.')[0])
        except ValueError:
            self.snap_num = 0
        with h5py.File(fname) as f:
            self.a = f["Header"].attrs["Time"]
            self.z = f["Header"].attrs["Redshift"]
            self.box_size = f["Header"].attrs["BoxSize"]
            self.softening = f["Parameters"].attrs["SofteningComovingClass0"]
            if self.softening * self.a > f["Parameters"].attrs["SofteningMaxPhysClass0"]:
                self.softening = f["Parameters"].attrs["SofteningMaxPhysClass0"] / self.a

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
    """Load GADGET-4 particle snapshots.

    Additional assumptions:
       only dark matter particles - PartType1
       all particles have the same mass

    Note: all data is loaded directly into memory,
      might be a problem for very large simulations.
    For a 256**3 particle sim in single precision a ParticleData object will use ~0.44GB
      a 512**3 particle sim will use ~3.5GB.

    Constructing trees as well will use significantly more memory:
    https://github.com/scipy/scipy/issues/15065.
    For a 512**3 sim, the tree will add around 5GB of memory usage.
    """

    def __init__(self, fname, load_vels=True, load_ids=True, make_tree=False):
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

            self.tree = None
            if make_tree:
                # make tree box size slightly larger to make sure it accomadates all the data
                tree_box = self.box_size * (1 + 1e-8)
                self.tree = KDTree(self.pos, boxsize=tree_box, leafsize=30)

    def query_radius(self, center, radius, count_only=False):
        """Return indicies of particles within radius of center."""
        if self.tree is None:
            warnings.warn("position tree not constructed, using brute force method.", RuntimeWarning)
            if radius > self.box_size:
                warnings.warn("radius is greater than box_size/2, periodic BC are not implemented with brute force method",
                        RuntimeWarning)
            mask = utils.get_sphere_mask(self.pos, center, radius)
            if count_only:
                return np.sum(mask)
            return np.nonzero(mask)[0]
        else:
            return self.tree.query_ball_point(center, radius, return_length=count_only)

    def query_ids(self, ids):
        """Return indicies of particles given list of ids."""
        if self.ids is None:
            raise RuntimeError("Cannot select particles, snapshot loaded with load_ids=False.")
        return np.nonzero(np.isin(self.ids, ids))[0]


class HaloCatalog(Snapshot):
    """Load GADGET-4 group snapshots (FoF and SUBFIND groups).
    """

    def __init__(self, fname, load_subhalos=False):
        super().__init__(fname)

        with h5py.File(fname) as f:
            self.n_halos = f["Header"].attrs["Ngroups_Total"]
            self.n_subhalos = 0

            if self.n_halos > 0:
                self.pos = f["Group"]["GroupPos"][:]
                self.vel = f["Group"]["GroupVel"][:]
                self.masses = f["Group"]["GroupMass"][:] * 1e10
                self.offsets = f["Group"]["GroupOffsetType"][:,1]
                self.lengths = f["Group"]["GroupLen"][:]

            if load_subhalos:
                try:
                    self.n_subhalos = f["Header"].attrs["Nsubhalos_Total"]
                    self.pos_sh = f["Subhalo"]["SubhaloPos"][:]
                    self.vel_sh = f["Subhalo"]["SubhaloVel"][:]
                    self.masses_sh = f["Subhalo"]["SubhaloMass"][:] * 1e10
                    self.offsets_sh = f["Subhalo"]["SubhaloOffsetType"][:,1]
                    self.lengths_sh = f["Subhalo"]["SubhaloLen"][:]
                except KeyError:
                    pass

    def get_particles(self, halo_i, subhalo=False):
        """Return indicies of particles that are part of the given halo."""
        offset = self.offsets_sh[halo_i] if subhalo else self.offsets[halo_i]
        length = self.lengths_sh[halo_i] if subhalo else self.lengths[halo_i]
        return np.arange(offset, offset+length)

    def get_particle_ids(self, halo_i, particle_data, subhalo=False):
        """Return ids of particles that are part of the given halo."""
        particles = self.get_particles(halo_i, subhalo=subhalo)
        return particle_data.ids[particles]

    def calc_fof_hmf(self, bins):
        """Calculate HMF of FoF clusters."""
        return utils.calc_hmf(bins, self.masses, self.box_size)

    def calc_subhalo_hmf(self, bins):
        """Calculate HMF of SUBFIND clusters."""
        return utils.calc_hmf(bins, self.masses_sh, self.box_size)
