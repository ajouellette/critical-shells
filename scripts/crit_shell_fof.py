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
    pd = ParticleData(snap_file, load_vels=False, make_tree=True)

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

    link_lens = link_params * pd.mean_particle_sep
    if rank == 0:
        print(f"Mean particle separation {pd.mean_particle_sep:.2f} Mpc")
        print(f"Using linking parameters {link_params}")

    n_fof = np.zeros((count[rank], len(link_params)), dtype=int)
    radii1 = np.zeros(count[rank])
    #frac_collapse = np.zeros(count[rank], dtype=float)
    #densities = np.zeros(count[rank])

    for i in range(count[rank]):
        center = all_centers[displ[rank] + i]
        radius = all_radii[displ[rank] + i]
        offset = sum(all_n[:displ[rank] + i])
        shell_ids = all_ids[offset:offset+all_n[displ[rank] + i]]

        # get particle positions at a=1
        rough_mask = pd.query_radius(center, 25)
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
        ind = pd.query_radius(center1, radius1)
        if len(ind) < 2:
            print("Error not enough particles")
            n_fof[i] = 0
            continue
        # center positions before running FoF
        pos_centered = center_box_pbc(pd.pos[ind], center1, pd.box_size)

        for link_i, link_len in enumerate(link_lens):
            groups = pyfof.friends_of_friends(np.double(pos_centered), link_len)
            group_lens = np.array([len(group) for group in groups])
            #main_group = np.argmax(group_lens)
            #print("number of groups {} size of main group {} size at a=100 {}".format(len(groups),
            #    len(groups[main_group]), len(shell_ids)), time.perf_counter() - time_start)

            # Find FoF group with the highest number of matching particles
            len_diffs = np.abs(group_lens - len(shell_ids))
            best_i = np.argmin(len_diffs)
            fof_ids = pd.ids[ind][groups[best_i]]
            best_count = np.sum(np.isin(shell_ids, fof_ids, assume_unique=True))
            for j in np.argsort(group_lens)[::-1]:
                ind_fof = groups[j]
                if len(ind_fof) < best_count:
                    break
                fof_ids = pd.ids[ind][ind_fof]
                match_count = np.sum(np.isin(shell_ids, fof_ids, assume_unique=True))
                if match_count == len(shell_ids):
                    best_i = j
                    best_count = match_count
                    break
                if match_count > best_count:
                    best_count = match_count
                    best_i = j

            #print(main_group, best_i, group_lens[best_i], best_count / len(shell_ids))
            main_group = best_i

            n_fof[i,link_i] = len(groups[main_group])

        # want percentage of particles within radius at a=1 that will collapse to halo at a=100
        #frac_collapse[i] = len(shell_ids) / len(ind)

        radii1[i] = radius1
        #densities[i] = density / crit_density
        #print(density/crit_density, "clipped" if radius1 < np.max(p_radii[mask_ids]) else "")

    comm.Barrier()

    if rank == 0:
        print("Finished.")
        all_n_fof = np.zeros((len(all_radii), len(link_lens)), dtype=int)
        all_radii1 = np.zeros_like(all_radii)
        #all_frac_collapse = np.zeros_like(all_radii, dtype=float)
        #all_densities = np.zeros_like(all_radii)
    else:
        all_n_fof = None
        all_radii1 = None
        #all_frac_collapse = None
        #all_densities = None

    comm.Gatherv(n_fof, [all_n_fof, count*len(link_lens), displ*len(link_lens),
        MPI.LONG], root=0)
    comm.Gatherv(radii1, [all_radii1, count, displ, MPI.DOUBLE], root=0)
    #comm.Gatherv(frac_collapse, [all_frac_collapse, count, displ, MPI.DOUBLE], root=0)
    #comm.Gatherv(densities, [all_densities, count, displ, MPI.DOUBLE], root=0)

    if rank == 0:
        data = {"n": all_n_fof, "link_params": link_params,
                "radii1": all_radii1}
        save_file = data_dir+"-analysis/fof_halos"
        print("Writing data to", save_file)
        with open(save_file, 'wb') as f:
            pickle.dump(data, f)
        print("Done.")


if __name__ == "__main__":
    main()
