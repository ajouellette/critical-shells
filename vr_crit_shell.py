import numpy as np
from mpi4py import MPI
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import pickle
import h5py
import time
from numba import jit
import snapshot

@jit(nopython=True)
def find_first(value, vector):
    """Find the index of the first occurence of value in vector."""
    for i, v in enumerate(vector):
        if value == v:
            return i
    return -1

@jit(nopython=True)
def find_first_npositive(vector):
    """Find index of first non-positive value in vector."""
    prod = 1
    for i, v in enumerate(vector):
        prod *= v
        if prod <= 0:
            break
    return i


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # read particle data on all processes
    data_dir = "/projects/caps/aaronjo2/dm-l256-n256-a100"
    particles = snapshot.ParticleData(data_dir + "/snapshot_021.hdf5")

    cosmo = FlatLambdaCDM(H0=particles.Hubble0, Om0=particles.OmegaMatter)

    if rank == 0:
        print("h = {:.3f}, a = {:.1f}".format(cosmo.h, particles.a))
        print("H = {:.2f}  {:.2f}".format(particles.Hubble, cosmo.H(-0.99)))

    # read all halo data on rank 0 and divide up work
    if rank == 0:
        with open(data_dir + "-analysis/spherical_halos_mpi", 'rb') as f:
            data = pickle.load(f)
        all_centers = data["centers"]
        all_radii = data["radii"]
        avg, res = divmod(len(all_radii), n_ranks)
        count = np.array([avg+1 if r < res else avg for r in range(n_ranks)])
        displ = np.array([sum(count[:r]) for r in range(n_ranks)])
    else:
        all_centers = None
        all_radii = None
        count = np.zeros(n_ranks, dtype=int)
        displ = 0

    comm.Bcast(count, root=0)

    centers = np.zeros((count[rank], 3))
    radii = np.zeros(count[rank])

    comm.Scatterv([all_radii, count, displ, MPI.DOUBLE], radii, root=0)
    comm.Scatterv([all_centers, 3*count, 3*displ, MPI.DOUBLE], centers, root=0)

    radii_vr = np.zeros_like(radii)

    for i in range(len(radii)):
        time_start = time.perf_counter()

        center = centers[i]
        radius = radii[i] / cosmo.h 
        r_cut = 1.5 * radius
        # calculate physical radial velocities of particles inside r_cut
        p_radii = np.linalg.norm(particles.pos - center, axis=1) / cosmo.h
        mask = p_radii < r_cut
        vel_center = np.mean(particles.vel[p_radii < radius], axis=0)
        pos_cut = (particles.pos[mask] - center) / cosmo.h
        vel_cut = np.sqrt(particles.a) * (particles.vel[mask] - vel_center)
        p_radii_cut = p_radii[mask]
        unit_vectors = pos_cut / np.expand_dims(p_radii_cut, 1)
        vr_peculiar = np.multiply(vel_cut, unit_vectors).sum(1)
        hubble_flow = particles.Hubble * particles.a * p_radii_cut
        vr_physical = vr_peculiar + hubble_flow
       
        time_mid = time.perf_counter()

        # find first radius after which vr_physical is no longer negative
        sorted_i = np.lexsort((vr_physical, p_radii_cut))
        index = find_first(0, np.cumprod(vr_physical[sorted_i][::-1] > 0))
        # average of last negative and next
        radii_vr[i] = 0.5 * (p_radii_cut[sorted_i][::-1][index] + p_radii_cut[sorted_i][::-1][index-1]) * cosmo.h

        time_end = time.perf_counter()

        print(i, radii_vr[i], " {:.4f} {:.4f}".format(time_mid - time_start, time_end - time_mid))

    comm.Barrier()
    if rank == 0:
        print("Done")
        all_radii_vr = np.zeros_like(all_radii)
    else:
        all_radii_vr = None
        
    comm.Gatherv(radii_vr, [all_radii_vr, count, displ, MPI.DOUBLE], root=0)
        
    if rank == 0:
        print("Gathered")
        with open(data_dir + "-analysis/vr_crit_radii", 'wb') as f:
            pickle.dump(all_radii_vr, f)

