import os
import sys
import h5py
import numpy as np
from mpi4py import MPI
import pyfof
from gadgetutils.snapshot import ParticleData
from gadgetutils.utils import center_box_pbc


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
    if not os.path.exists(snap_file):
        if rank == 0:
            print("Error: could not find data file")
        sys.exit(1)

    # read particle data on all processes
    pd = ParticleData(snap_file, load_vels=False, make_tree=True)

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

    n_fof = np.zeros_like(radii, dtype=int)
    ids_fof = []

    # FoF linking length
    link_len = 0.2 * pd.mean_particle_sep
    if rank == 0:
        print(f"Using FoF linking length {link_len:.2f} Mpc")

    for i in range(len(radii)):
        center = centers[i]
        radius = radii[i]
        # get particles within shell and re-center
        mask = pd.tree.query_ball_point(center, 2*radius)
        pos_cut = pd.pos[mask]
        pos_cut = center_box_pbc(pos_cut, center, pd.box_size)
        # run FoF
        if len(pos_cut) < 2:
            print("Error not enough particles")
            continue
        groups = pyfof.friends_of_friends(np.double(pos_cut), link_len)
        group_lens = [len(groups[i]) for i in range(len(groups))]
        main_group = np.argmax(group_lens)
        #print("number of groups {} size of main group {} size of crit shell {}".format(len(groups), len(groups[main_group]), np.sum(p_radii<radius)))

        n_fof[i] = len(groups[main_group])
        ids_fof = ids_fof + groups[main_group]

    comm.Barrier()

    ids_fof = np.array(ids_fof)
    length = np.array(len(ids_fof))
    total_length = np.array(0)
    comm.Reduce(length, total_length, op=MPI.SUM, root=0)
    lengths = np.array(comm.gather(length, root=0))
    displ_ids = 0

    comm.Barrier()
    if rank == 0:
        print("Finished")
        all_n_fof = np.zeros_like(all_radii, dtype=int)
        all_ids_fof = np.zeros(total_length, dtype=int)
        displ_ids = np.array([sum(lengths[:r]) for r in range(n_ranks)])
    else:
        all_n_fof = None
        all_ids_fof = None

    comm.Gatherv(n_fof, [all_n_fof, count, displ, MPI.LONG], root=0)
    comm.Gatherv(ids_fof, [all_ids_fof, lengths, displ_ids, MPI.LONG], root=0)

    if rank == 0:
        save_file = data_dir + "-analysis/fof_halos_100.hdf5"
        print("Writing data to ", save_file)
        with h5py.File(save_file, 'w') as f:
            f.attrs["Nshells"] = len(all_n_fof)
            f.create_dataset("Nparticles", data=all_n_fof)
            f.create_dataset("ParticleIDs", data=all_ids_fof)

        print("Done.")


if __name__ == "__main__":
    main()
