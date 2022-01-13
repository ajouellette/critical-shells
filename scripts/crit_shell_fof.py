import os
import sys
import h5py
import pickle
import numpy as np
from mpi4py import MPI
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import pyfof
from gadgetutils.snapshot import ParticleData
from gadgetutils.utils import mean_pos_pbc, center_box_pbc

import time


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    if len(sys.argv) != 3:
        if rank == 0:
            print("Error: need data dir and snapshot number")
        sys.exit(1)

    data_dir = sys.argv[1]
    snap_num1 = sys.argv[2]
    snap_file1 = data_dir + "/snapshot_" + snap_num1 + ".hdf5"
    if rank == 0:
        print("Using data files {}".format(snap_file1))
    if not os.path.exists(snap_file1):
        if rank == 0:
            print("Error: could not find data files")
        sys.exit(1)

    # read particle data on all processes
    pd = ParticleData(snap_file1, load_vels=False, make_tree=True)

    cosmo = FlatLambdaCDM(Om0=pd.OmegaMatter, H0=100)
    crit_density = cosmo.critical_density0.to(u.Msun / u.Mpc**3).value

    # critical shell data
    with h5py.File(data_dir + "-analysis/critical_shells.hdf5") as f:
        all_centers = f["Centers"][:]
        all_radii = f["Radii"][:]
        all_n = f["Nparticles"][:]
        all_ids = f["ParticleIDs"][:]

    avg, res = divmod(len(all_radii), n_ranks)
    count = np.array([avg+1 if r < res else avg for r in range(n_ranks)])
    displ = np.array([sum(count[:r]) for r in range(n_ranks)])

    n_fof = np.zeros(count[rank], dtype=int)
    frac_collapse = np.zeros(count[rank], dtype=float)
    radii1 = np.zeros(count[rank])
    densities = np.zeros(count[rank])
    ids_fof = []

    link_len = 0.2 * pd.box_size / pd.n_parts**(1/3)
    if rank == 0:
        print(f"Using linking length {link_len} Mpc")

    for i in range(count[rank]):
        time_start = time.perf_counter()
        center = all_centers[displ[rank] + i]
        radius = all_radii[displ[rank] + i]
        offset = sum(all_n[:displ[rank] + i])
        shell_ids = all_ids[offset:offset+all_n[displ[rank] + i]]

        # get particle positions at a=1
        rough_mask = pd.tree.query_ball_point(center, 25)
        ind_ids = np.nonzero(np.isin(pd.ids[rough_mask], shell_ids))
        pos_shell = pd.pos[rough_mask][ind_ids]

        # double check that we actually got all particles
        if len(pos_shell) != len(shell_ids):
            print("missed some particles", len(pos_shell), len(shell_ids))

        center1 = mean_pos_pbc(pos_shell, pd.box_size)

        # recenter box on CoM
        pos_shell_centered = center_box_pbc(pos_shell, center1, pd.box_size)

        p_radii = np.linalg.norm(pos_shell_centered, axis=1)
        radius1 = np.max(p_radii)
        #density = np.sum(mask_r1) * part_mass / (4/3 * np.pi * radius1**3)
        #print("a=100: {} {:.4f}   a=1: {} {:.4f}".format(center, radius, center1, radius1))

        # run FoF
        ind = pd.tree.query_ball_point(center1, radius1)
        if len(ind) < 2:
            print("Error not enough particles")
            n_fof[i] = 0
            continue
        # center positions before running FoF
        pos_centered = center_box_pbc(pd.pos[ind], center1, pd.box_size)

        groups = pyfof.friends_of_friends(np.double(pos_centered), link_len)
        group_lens = [len(group) for group in groups]
        main_group = np.argmax(group_lens)
        print("number of groups {} size of main group {} size at a=100 {}".format(len(groups),
            len(groups[main_group]), len(shell_ids)), time.perf_counter() - time_start)

        # Find FoF group with the highest number of matching particles
        best_count = 0
        best_i = np.argmax(group_lens)
        for j in np.argsort(group_lens)[::-1][:10]:
            ind_fof = groups[j]
            fof_ids = pd.ids[ind][ind_fof]
            match_count = np.sum(np.isin(shell_ids, fof_ids))
            if match_count > best_count:
                best_count = match_count
                best_i = j

        print(main_group, best_i, group_lens[best_i])
        main_group = best_i

        # want percentage of particles within radius at a=1 that will collapse to halo at a=100
        frac_collapse[i] = len(shell_ids) / len(ind)

        n_fof[i] = len(groups[main_group])
        ids_fof = ids_fof + groups[main_group]

        radii1[i] = radius1
        #densities[i] = density / crit_density
        #print(density/crit_density, "clipped" if radius1 < np.max(p_radii[mask_ids]) else "")

    comm.Barrier()

    ids_fof = np.array(ids_fof)
    length = np.array(len(ids_fof))
    total_length = np.array(0)
    comm.Reduce(length, total_length, op=MPI.SUM, root=0)
    lengths = np.array(comm.gather(length, root=0))
    displ_ids = 0

    comm.Barrier()
    if rank == 0:
        print("Done")
        all_n_fof = np.zeros_like(all_radii, dtype=int)
        all_ids_fof = np.zeros(total_length, dtype=int)
        all_frac_collapse = np.zeros_like(all_radii, dtype=float)
        all_radii1 = np.zeros_like(all_radii)
        all_densities = np.zeros_like(all_radii)
        displ_ids = np.array([sum(lengths[:r]) for r in range(n_ranks)])
    else:
        all_n_fof = None
        all_ids_fof = None
        all_frac_collapse = None
        all_radii1 = None
        all_densities = None

    comm.Gatherv(n_fof, [all_n_fof, count, displ, MPI.LONG], root=0)
    comm.Gatherv(ids_fof, [all_ids_fof, lengths, displ_ids, MPI.LONG], root=0)
    comm.Gatherv(frac_collapse, [all_frac_collapse, count, displ, MPI.DOUBLE], root=0)
    comm.Gatherv(radii1, [all_radii1, count, displ, MPI.DOUBLE], root=0)
    comm.Gatherv(densities, [all_densities, count, displ, MPI.DOUBLE], root=0)

    if rank == 0:
        print("Gathered")
        data = {"n": all_n_fof, "ids": all_ids_fof, "frac": all_frac_collapse,
                "radii1": all_radii1, "densities": all_densities}
        with open(data_dir+"-analysis/fof_halos", 'wb') as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    main()
