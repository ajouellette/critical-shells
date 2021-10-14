import numpy as np
from mpi4py import MPI
import pickle
import h5py
import pyfof
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import os
import sys
import snapshot


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    if len(sys.argv) != 4:
        if rank == 0:
            print("Error: need data dir and snapshot numbers")
        sys.exit(1)

    data_dir = sys.argv[1]
    snap_num1 = sys.argv[2]
    snap_num100 = sys.argv[3]
    snap_file1 = data_dir + "/snapshot_" + snap_num1 + ".hdf5"
    snap_file100 = data_dir + "/snapshot_" + snap_num100 + ".hdf5"
    if rank == 0:
        print("Using data files {} and {}".format(snap_file1, snap_file100))
    if not (os.path.exists(snap_file1) and os.path.exists(snap_file100)):
        if rank == 0:
            print("Error: could not find data files")
        sys.exit(1)

    # read particle data on all processes
    with h5py.File(snap_file100) as f:  # a = 100
        pos100 = f["PartType1"]["Coordinates"][:]
        ids100 = f["PartType1"]["ParticleIDs"][:]
        box_size = f["Header"].attrs["BoxSize"]
        part_mass = f["Header"].attrs["MassTable"][1] * 1e10
    with h5py.File(snap_file1) as f:  # a = 1
        pos1 = f["PartType1"]["Coordinates"][:]
        ids1 = f["PartType1"]["ParticleIDs"][:]
        cosmo = FlatLambdaCDM(Om0=f["Parameters"].attrs["Omega0"], H0=100)
        crit_density = cosmo.critical_density0.to(u.Msun / u.Mpc**3).value

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

    n_fof = np.zeros_like(radii, dtype=int)
    frac_collapse = np.zeros_like(radii, dtype=float)
    radii1 = np.zeros_like(radii)
    densities = np.zeros_like(radii)
    ids_fof = []

    for i in range(len(radii)):
        center = centers[i]
        radius = radii[i] 
        # get ids within shell at a=100
        p_radii = np.linalg.norm(pos100 - center, axis=1)
        ids = ids100[p_radii < radius]
        # get particle positions at a=1
        mask_ids = np.isin(ids1, ids)
        pos_shell = pos1[mask_ids]
        # calculate center taking into account periodic BC
        theta = 2*np.pi * pos_shell / box_size
        # use median to calculate CoM? - more robust against outliers which we remove later
        # do we need to remove outliers?? - might shrink radius too much
        #center1 = (np.arctan2(np.mean(-np.sin(theta), axis=0), -np.mean(np.cos(theta), axis=0)) + np.pi) * 256/(2*np.pi)
        center1 = (np.arctan2(np.median(-np.sin(theta), axis=0), -np.median(np.cos(theta), axis=0)) + np.pi) * box_size/(2*np.pi)
        # recenter box on CoM
        dx = center1 + box_size*(center1 < -box_size/2) - box_size*(center1 >= box_size/2)
        pos1_centered = pos1 - dx
        pos1_centered = pos1_centered + box_size*(pos1_centered < -box_size/2) - box_size*(pos1_centered >= box_size/2)
        p_radii = np.linalg.norm(pos1_centered, axis=1)
        # sigmaclip to remove outliers - some particles move way further than expected when traced back
        # tune value of high? (maybe need smaller value)
        radius1 = np.max(stats.sigmaclip(p_radii[mask_ids], high=3.5)[0])
        mask_r1 = p_radii < radius1
        density = np.sum(mask_r1) * part_mass / (4/3 * np.pi * radius1**3)
        print("a=100: {} {:.4f}   a=1: {} {:.4f}".format(center, radius, center1, radius1))
        # run FoF
        link_len = 0.2
        if np.sum(mask_r1) < 2:
            print("Error not enough particles")
            n_fof[i] = 0
            continue
        groups = pyfof.friends_of_friends(np.double(pos1_centered[mask_r1]), link_len)
        group_lens = [len(groups[i]) for i in range(len(groups))]
        main_group = np.argmax(group_lens)
        print("number of groups {} size of main group {} size at a=100 {}".format(len(groups), len(groups[main_group]), len(ids)))

        # want percentage of particles within radius at a=1 that will collapse to halo at a=100
        frac_collapse[i] = np.sum(mask_ids) / np.sum(mask_r1)

        n_fof[i] = len(groups[main_group])
        ids_fof = ids_fof + groups[main_group]

        radii1[i] = radius1
        densities[i] = density / crit_density
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
        data = {"n":all_n_fof, "ids":all_ids_fof, "frac":all_frac_collapse, "radii1":all_radii1, "densities":all_densities}
        with open(data_dir+"-analysis/fof_halos", 'wb') as f:
            pickle.dump(data, f)

