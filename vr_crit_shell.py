import os
import sys
import pickle
import numpy as np
import numba as nb
from mpi4py import MPI
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from sklearn import neighbors
import h5py
import snapshot
from utils import mean_pos

import time

@nb.njit
def find_first(value, vector):
    """Find the index of the first occurence of value in vector."""
    for i, v in enumerate(vector):
        if value == v:
            return i
    return -1

@nb.njit
def find_first_npositive(vector):
    """Find index of first non-positive value in vector."""
    prod = 1
    for i, v in enumerate(vector):
        prod *= v
        if prod <= 0:
            break
    return i


@nb.njit
def calc_vr_phys(pos, vel, center, radius, a, H, h):
    pos_centered = pos - center
    p_radii = np.sqrt(np.sum(pos_centered**2, axis=1))
    vel_center = mean_pos(vel[p_radii < radius])
    vel_peculiar = np.sqrt(a) * (vel - vel_center)
    unit_vectors = pos_centered / np.expand_dims(p_radii, 1)
    vr_peculiar = np.multiply(vel_peculiar, unit_vectors).sum(1)
    hubble_flow = H * a * p_radii / h
    vr_physical = vr_peculiar + hubble_flow
    return p_radii, vr_physical


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    if len(sys.argv) != 3:
        if rank == 0:
            print("Error: need data dir and snapshot number")
        sys.exit(1)

    data_dir = sys.argv[1]
    snap_num = sys.argv[2]
    snap_file = data_dir + "/snapshot_" + snap_num + ".hdf5"
    if not os.path.exists(snap_file):
        if rank == 0:
            print("Error: could not find data file")
        sys.exit(1)
    # read particle data on all processes
    pd = snapshot.ParticleData(snap_file)
    pos_tree = neighbors.BallTree(pd.pos)

    comm.Barrier()
    if rank == 0:
        print("Memory usage after loading particle data and constructing trees")
        os.system("free -h")

    cosmo = FlatLambdaCDM(H0=pd.Hubble0, Om0=pd.OmegaMatter)

    if rank == 0:
        print("h = {:.3f}, a = {:.1f}".format(cosmo.h, pd.a))
        print("H = {:.2f}  {:.2f}".format(pd.Hubble, cosmo.H(-0.99)))

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
        radius = radii[i]
        r_cut = 2 * radius
        mask = pos_tree.query_radius(center.reshape(1,-1), r_cut)[0]
        p_radii_cut, vr_physical = calc_vr_phys(pd.pos[mask], pd.vel[mask], center, radius, pd.a, pd.Hubble, cosmo.h)

        # find first radius after which vr_physical is no longer negative
        sorted_i = np.lexsort((vr_physical, p_radii_cut))
        index = find_first(0, np.cumprod(vr_physical[sorted_i][::-1] > 0))
        # average of last negative and next
        radii_vr[i] = 0.5 * (p_radii_cut[sorted_i][::-1][index] + p_radii_cut[sorted_i][::-1][index-1])

        time_end = time.perf_counter()

        print(i, radii_vr[i], "  {:.4f} sec".format(time_end - time_start))

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

