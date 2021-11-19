import os
import sys
import h5py
import numpy as np
import numba as nb
from mpi4py import MPI
from astropy.cosmology import FlatLambdaCDM
from sklearn import neighbors
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
    """Calculate physical radial velocities."""
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
    pd = snapshot.ParticleData(snap_file, load_ids=False)
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
        with h5py.File(data_dir + "-analysis/critical_shells.hdf5") as f:
            all_centers = f["Centers"][:]
            all_radii = f["Radii"][:]
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
    n_parts = np.zeros_like(radii, dtype=int)
    avg_vr = np.zeros_like(radii)
    std_vr = np.zeros_like(radii)

    for i in range(len(radii)):
        time_start = time.perf_counter()

        center = centers[i]
        radius = radii[i]
        # calculate physical vr for particles in/near shell
        r_cut = 2 * radius
        mask = pos_tree.query_radius(center.reshape(1, -1), r_cut)[0]
        pos_cut = pd.pos[mask]
        vel_cut = pd.vel[mask]
        p_radii_cut, vr_physical = calc_vr_phys(pos_cut, vel_cut, center, radius, pd.a, pd.Hubble, cosmo.h)

        # find first radius after which vr_physical is no longer negative
        sorted_i = np.lexsort((vr_physical, p_radii_cut))
        index = find_first(0, np.cumprod(vr_physical[sorted_i][::-1] > 0))
        # average of last negative and next
        radius_vr = 0.5 * (p_radii_cut[sorted_i][::-1][index] + p_radii_cut[sorted_i][::-1][index-1])

        # number of particles inside radius_vr
        n_vr = pos_tree.query_radius(center.reshape(1, -1), radius_vr, count_only=True)

        # calculate vr quantities inside shell
        mask_shell = p_radii_cut < radius
        vr_phys_shell = vr_physical[mask_shell]

        avg_vr[i] = np.mean(vr_phys_shell)
        std_vr[i] = np.std(vr_phys_shell)
        radii_vr[i] = radius_vr
        n_parts[i] = n_vr

        time_end = time.perf_counter()

        print((rank, i), radii_vr[i], "  {:.4f} sec".format(time_end - time_start))

    comm.Barrier()
    if rank == 0:
        print("Finished calculations")
        all_radii_vr = np.zeros_like(all_radii)
        all_n_parts = np.zeros_like(all_radii, dtype=int)
        all_avg_vr = np.zeros_like(all_radii)
        all_std_vr = np.zeros_like(all_radii)
    else:
        all_radii_vr = None
        all_n_parts = None
        all_avg_vr = None
        all_std_vr = None

    comm.Gatherv(radii_vr, [all_radii_vr, count, displ, MPI.DOUBLE], root=0)
    comm.Gatherv(n_parts, [all_n_parts, count, displ, MPI.LONG], root=0)
    comm.Gatherv(avg_vr, [all_avg_vr, count, displ, MPI.DOUBLE], root=0)
    comm.Gatherv(std_vr, [all_std_vr, count, displ, MPI.DOUBLE], root=0)

    if rank == 0:
        save_file = data_dir + "-analysis/vr_data.hdf5"
        print("Writing data to ", save_file)
        with h5py.File(save_file, 'w') as f:
            f.attrs["Nshells"] = len(all_radii_vr)
            f.create_dataset("Radii", data=all_radii_vr)
            f.create_dataset("Nparticles", data=all_n_parts)
            f.create_dataset("AvgVr", data=all_avg_vr)
            f.create_dataset("StdVr", data=all_std_vr)

        print("Done.")

