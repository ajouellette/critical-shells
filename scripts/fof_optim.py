import os
import sys
import h5py
import pickle
import numpy as np
from mpi4py import MPI
from scipy import spatial
import pyfof
from gadgetutils.snapshot import ParticleData
from gadgetutils.utils import mean_pos_pbc, center_box_pbc

import time

link_params = np.linspace(0.05, 0.6, 31)


def main():
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
    if rank == 0:
        print("Using data files {}".format(snap_file))
    if not os.path.exists(snap_file):
        if rank == 0:
            print("Error: could not find data files")
        sys.exit(1)

    # read particle data on all processes
    pd = ParticleData(snap_file, load_vels=False)
    # hack to convert positions in range (0,L] from GADGET to range [0,L) for the KDTree
    pos_tree = spatial.KDTree(np.float64(pd.pos) - np.min(pd.pos),
            boxsize=pd.box_size, leafsize=30)

    # critical shell data
    with h5py.File(data_dir + "-analysis/critical_shells.hdf5") as f:
        all_centers = f["Centers"][:]
        all_radii = f["Radii"][:]
        all_n = f["Nparticles"][:]
        all_ids = f["ParticleIDs"][:]

    avg, res = divmod(len(all_radii), n_ranks)
    count = np.array([avg+1 if r < res else avg for r in range(n_ranks)])
    displ = np.array([sum(count[:r]) for r in range(n_ranks)])

    link_lens = link_params * pd.mean_particle_sep
    if rank == 0:
        print(f"Mean particle separation {pd.mean_particle_sep:.2f} Mpc")
        print(f"Using linking lengths {link_lens} Mpc")

    n_fof = np.zeros((count[rank], len(link_lens)), dtype=int)

    for i in range(count[rank]):
        time_start = time.perf_counter()
        center = all_centers[displ[rank] + i]
        radius = all_radii[displ[rank] + i]
        offset = sum(all_n[:displ[rank] + i])
        shell_ids = all_ids[offset:offset+all_n[displ[rank] + i]]

        # get particle positions at a=1
        rough_mask = pos_tree.query_ball_point(center, 25)
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

        # run FoF
        ind = pos_tree.query_ball_point(center1, radius1)
        if len(ind) < 2:
            print("Error not enough particles")
            n_fof[i] = 0
            continue
        # center positions before running FoF
        pos_centered = center_box_pbc(pd.pos[ind], center1, pd.box_size)

        for link_i, link_len in enumerate(link_lens):
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

            n_fof[i,link_i] = len(groups[main_group])

    comm.Barrier()

    if rank == 0:
        print("Done")
        all_n_fof = np.zeros((len(all_radii), len(link_lens)), dtype=int)
    else:
        all_n_fof = None

    comm.Gatherv(n_fof, [all_n_fof, len(link_lens) * count, len(link_lens) * displ, MPI.LONG], root=0)

    if rank == 0:
        print("Gathered")
        data = {"n": all_n_fof, "link_params": link_params}
        with open(data_dir+"-analysis/fof_optim", 'wb') as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    main()
