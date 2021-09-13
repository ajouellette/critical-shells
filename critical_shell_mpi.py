import numpy as np
from mpi4py import MPI
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import pickle
import h5py
import time
import os
import sys
import snapshot


def get_density(radii, cut_radius, part_mass, a=1):
    num_particles = np.sum(radii < cut_radius)
    volume = 4/3 * np.pi * (cut_radius * a)**3
    return num_particles * part_mass / volume


def get_mass(radius, density, a=1):
    volume = 4/3 * np.pi * (radius * a)**3
    return density * volume


def critical_shell(pos, center, part_mass, crit_dens, crit_ratio=2, center_tol=1e-3, rcrit_tol=5e-4, maxiters=100):
    """Find a critical shell starting from given center."""
    radius = 0.1
    # find center of largest substructure
    center_converged = False
    for i in range(20):
        radii = np.sqrt(np.sum((pos - center)**2, axis=1))
        new_center = np.mean(pos[radii < radius], axis=0)
        if np.linalg.norm(center - new_center) < center_tol:
            center_converged = True
            break
        center = new_center
        radius += 0.05

    # decrease radius and then grow to get critical shell
    radius = 1e-3
    radii = np.sqrt(np.sum((pos - center)**2, axis=1))
    dr = rcrit_tol * 100
    iters = i
    density_converged = False
    while dr > rcrit_tol:
        iters += 1
        new_center = center + np.mean(pos[radii < radius + dr] - center, axis=0)
        new_radii = np.sqrt(np.sum((pos - new_center)**2, axis=1))
        density = get_density(new_radii, radius+dr, part_mass, a=100)

        if density < crit_ratio * crit_dens:
            dr /= 2
        else:
            center = new_center
            radii = new_radii
            radius += dr
        if iters > maxiters:
            break
    
    # convergence check (maybe not necessary)
    if density < crit_ratio * crit_dens and density > crit_dens and dr < rcrit_tol:
        density_converged = True

    return center, radius, center_converged, density_converged



if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    
    if rank == 0:
        print("running on {} cores".format(n_ranks))

    pd = snapshot.ParticleData("/projects/caps/aaronjo2/dm-l256-n256-a100/snapshot_021.hdf5")
    #hc = snapshot.HaloCatalog("/home/aaron/dm-l50-n128-a100/fof_tab_012.hdf5")
    fof_tab_data = h5py.File("/projects/caps/aaronjo2/dm-l256-n256-a100/fof_subhalo_tab_021.hdf5", 'r')
    # 2**15 = 32768
    sh_masses = fof_tab_data["Subhalo"]["SubhaloMass"][:32768] * 1e10
    sh_pos = fof_tab_data["Subhalo"]["SubhaloCM"][:32768]

    group_offsets = fof_tab_data["Group"]["GroupOffsetType"][:,1]
    group_lens = fof_tab_data["Group"]["GroupLen"][:]
    group_ends = group_offsets + group_lens
    outer_fuzz_start = group_ends[-1]

    sh_group_i = fof_tab_data["Subhalo"]["SubhaloGroupNr"][:]

    fof_tab_data.close()


    # Use H0 = 100 to factor out h
    cosmo = FlatLambdaCDM(H0=100, Om0=pd.OmegaMatter)
    crit_dens_a100 = cosmo.critical_density(-0.99).to(u.Msun / u.Mpc**3).value

    radii = []
    centers_s = []

    per_rank = int(len(sh_masses) / n_ranks)
    
    if rank == 0:
        print("Critical density at a = 100: {:.3e}".format(crit_dens_a100))
        print("Particle mass: {:.3e}".format(pd.part_mass))
        print("{} subhalos to process, {} per rank".format(len(sh_masses), per_rank))

    check_dup_len = 10

    halos_found = 0
    sh_processed = 0
    time_start = time.perf_counter()
    for i in range(rank * per_rank, (rank+1) * per_rank):

        print("rank {}, i {}  mass {:.2e}".format(rank, i, sh_masses[i]))

        sh_processed += 1

        if sh_processed % 100 == 0 and sh_processed != 0:
            time_end = time.perf_counter()
            print(sh_processed, halos_found, "{:.2f} sec".format(time_end - time_start))
            time_start = time_end

        center = sh_pos[i]

        # ignore centers too close to box boundary
        if np.sum(center < 2) or np.sum(center > pd.box_size - 2):
            print("Skipping:", center, "too close to boundary")
            continue

        # only use subset of all positions to speed up critical shell search
        group_i = sh_group_i[i]
        pos_indicies = np.hstack((np.arange(group_offsets[group_i], group_ends[group_i]), 
            np.arange(outer_fuzz_start, len(pd.pos))))

        center_s, radius, c_converged, d_converged = critical_shell(pd.pos[pos_indicies], center, 
                pd.part_mass, crit_dens_a100)
        if np.sum(center_s < 2) or np.sum(center_s > pd.box_size - 2):
            print("Skipping:", center, "too close to boundary")
            continue

        if c_converged and d_converged:
            # check for duplicates on same node, still need to check for duplicates
            #  across all nodes
            duplicate = False
            for i in range(np.max((0, len(centers_s) - check_dup_len)), len(centers_s)):
                if np.linalg.norm(centers_s[i] - center_s) < radii[i] + radius:
                    print("r {} i {}  Skipping duplicate:".format(rank,i), centers_s[i], center_s)
                    duplicate = True
                    break
            if not duplicate:
                print("Found halo:", center, radius)
                radii.append(radius)
                centers_s.append(center_s)
                halos_found += 1
        else:
            print("Did not converge:", "center" if not c_converged else '',
                    "density" if not d_converged else '')

    
    centers = np.array(centers_s, dtype=np.float64)
    radii = np.array(radii, dtype=np.float64)
    length = np.array(len(centers))

    total_length = np.array(0)
    comm.Reduce(length, total_length, op=MPI.SUM, root=0)

    lengths_c = np.array(comm.gather(3 * length, root=0))
    lengths_r = np.array(comm.gather(length, root=0))

    displ_c = None
    displ_r = None
    recvbuf_c = None
    recvbuf_r = None
    if rank == 0:
        displ_c = np.array([sum(lengths_c[:r]) for r in range(n_ranks)])
        displ_r = np.array([sum(lengths_r[:r]) for r in range(n_ranks)])
        recvbuf_c = np.zeros((total_length,3))
        recvbuf_r = np.zeros(total_length)

    comm.Barrier()

    # Gather all data together on node 0
    comm.Gatherv(centers, [recvbuf_c, lengths_c, displ_c, MPI.DOUBLE], root=0)
    comm.Gatherv(radii, [recvbuf_r, lengths_r, displ_r, MPI.DOUBLE], root=0)

    if rank == 0:
        # need to check for duplicates
        print("Checking for duplicates")


        print("Finished, {} spherical halos found".format(len(recvbuf_r)))
        masses = get_mass(recvbuf_r, crit_dens_a100, a=100)
        data = {"centers":recvbuf_c, "radii":recvbuf_r, "masses":masses}
        #fname_min = "None" if not min_mass else "{:.1e}".format(min_mass).replace('+','') 
        #fname_max = "None" if not max_mass else "{:.1e}".format(max_mass).replace('+','')
        file = open("spherical_halos_mpi", "wb")
        pickle.dump(data, file)
        file.close()

