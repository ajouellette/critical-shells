import os
import sys
import numpy as np
from mpi4py import MPI
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import h5py
from gadgetutils.snapshot import ParticleData
from gadgetutils.utils import sphere_volume, mean_pos_pbc

import time
profile = False


def get_density(pd, center, radius):
    """Get average enclosed density of a sphere from a ParticleData instance."""
    n = pd.query_radius(center, radius, count_only=True)
    return n * pd.part_mass / sphere_volume(radius, a=pd.a)


def find_critical_shell(pd, center, crit_dens, crit_ratio=2, center_tol=1e-3,
        rcrit_tol=1e-5, maxiters=100, findCOM=True):
    """Find a critical shell starting from given center.

    Parameters:
    pd: a ParticleData instance
    center: ndarray (1,3) - Initial guess for center of shell
    crit_dens: float - Critical density of universe
    crit_ratio: float - Ratio of critical shell density to critical density
    center_tol: float, optional - Tolerance for center convergence
    rcrit_tol: float, optional - Tolerance for radius convergence
    maxiters: int, optional - Maximum number of iterations

    Returns:
    center: ndarray (1,3) - Center of shell
    radius: float - Radius of shell
    n_parts: int - Number of particles enclosed
    center_converged: bool
    density_converged: bool
    """
    center_converged = True
    if findCOM:
        radius = 5e-3
        center_converged = False
        # find center of largest substructure
        for iters in range(1, 20):
            ind = pd.query_radius(center, radius)
            if len(ind) == 0:
                radius += 5e-3
                continue
            new_center = mean_pos_pbc(pd.pos[ind], pd.box_size)
            if np.linalg.norm(center - new_center) < center_tol:
                center_converged = True
                break
            center = new_center
            radius += 5e-3
    else:
        iters = 0

    # initial bracket for radius of critical shell
    r_low = 5e-4
    r_high = 0.5
    density_low = get_density(pd, center, r_low)
    # potentially need a better lower bound if no particles within initial radius
    while density_low == 0:
        r_low += 5e-4
        density_low = get_density(pd, center, r_low)
    density_high = get_density(pd, center, r_high)

    # check that guess actually brackets root
    if not (density_low / crit_dens > crit_ratio and
            density_high / crit_dens < crit_ratio):
        print("error initial guesses not good enough",
                (r_low, density_low / crit_dens), (r_high, density_high / crit_dens))
        return center, 0, 0, center_converged, False

    density_converged = False
    for i in range(iters+1, maxiters):
        r_mid = (r_low + r_high) / 2
        ind = pd.query_radius(center, r_mid)
        if len(ind) == 0:
            break  # ideally shouldn't happen, some problem with convergence
        center = mean_pos_pbc(pd.pos[ind], pd.box_size)
        density_mid = get_density(pd, center, r_mid)

        if density_mid / crit_dens == crit_ratio:
            density_converged = True
            break

        val_low = density_low / crit_dens - crit_ratio
        val_mid = density_mid / crit_dens - crit_ratio
        val_high = density_high / crit_dens - crit_ratio
        if val_low * val_mid < 0:
            r_high = r_mid
            density_high = density_mid
        elif val_mid * val_high < 0:
            r_low = r_mid
            density_low = density_mid
        else:
            print("convergence error, this should not happen")
            return center, r_mid, 0, center_converged, False

        if (r_high - r_low) / 2 < rcrit_tol:
            density_converged = True
            break

    # number of particles
    n_parts = pd.query_radius(center, r_mid, count_only=True)

    return center, r_mid, n_parts, center_converged, density_converged


def check_duplicate(centers, radii, center, radius=0, check_dup_len=10):
    """Check if a shell is a duplicate or contained within another shell."""
    for i in range(np.max((0, len(centers) - check_dup_len)), len(centers)):
        if np.linalg.norm(centers[i] - center) < radii[i] + radius:
            return i
    return -1


def print_conv_error(c_conv, d_conv, fof_i, sh_i=0):
    print("Did not converge ({}, {}):".format(fof_i, sh_i),
            "center" if not c_conv else '', "density" if not d_conv else '')


def print_dup_error(fof_i, sh_i, center, radius):
    print("Skipping duplicate: fof {}, sh {} ".format(fof_i, sh_i), center, radius)


def main():
    if profile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    np.random.seed(10)

    min_n_particles = 10

    if len(sys.argv) != 3:
        if rank == 0:
            print("Error: need data dir and snapshot number")
        sys.exit(1)

    data_dir = sys.argv[1]
    snap_num = sys.argv[2]
    snap_file = data_dir + "/snapshot_"+snap_num+".hdf5"
    fof_file = data_dir + "/fof_subhalo_tab_"+snap_num+".hdf5"
    if not (os.path.exists(snap_file) and os.path.exists(fof_file)):
        if rank == 0:
            print("Error: data files not found")
        sys.exit(1)

    if rank == 0:
        print(f"Using data file {snap_file}")
        print(f"And FoF group catalog {fof_file}")
    time_start = time.perf_counter()
    # load particle data and construct tree of positions
    pd = ParticleData(snap_file, load_vels=False, make_tree=True)

    comm.Barrier()
    time_end = time.perf_counter()
    if rank == 0:
        print(f"Time to load data and construct trees {(time_end - time_start)/60:.2f}"
              + " minutes")
        print("Memory usage after tree construction:")
        os.system("free -h")

    # Use H0 = 100 to factor out h, calculate critical density
    cosmo = FlatLambdaCDM(H0=100, Om0=pd.OmegaMatter)
    crit_dens_a100 = cosmo.critical_density(-0.99).to(u.Msun / u.Mpc**3).value

    # load FoF group data
    with h5py.File(fof_file) as fof_tab_data:
        if rank == 0:
            load = 2000 if profile else fof_tab_data["Header"].attrs["Ngroups_Total"]
            fof_pos_all = fof_tab_data["Group"]["GroupPos"][:load]
            fof_sh_num_all = fof_tab_data["Group"]["GroupNsubs"][:load]

        comm.Barrier()

        sh_pos = fof_tab_data["Subhalo"]["SubhaloPos"][:]
        sh_offsets = fof_tab_data["Group"]["GroupFirstSub"][:]

    if rank == 0:
        # some debug info
        print()
        print(f"a = {pd.a:.2f}")
        print(f"Critical density at a = 100: {crit_dens_a100:.3e}")
        print(f"Particle mass: {pd.part_mass:.3e}")
        print(f"Processing {len(fof_pos_all)} FoF groups on {n_ranks} MPI ranks" +
                f", {len(fof_pos_all)//n_ranks} per rank")

        avg, res = divmod(len(fof_pos_all), n_ranks)
        count = np.array([avg+1 if r < res else avg for r in range(n_ranks)])
        displ = np.array([sum(count[:r]) for r in range(n_ranks)])
        # randomize data before scattering to make load more even
        shuffle = np.arange(len(fof_pos_all))
        np.random.shuffle(shuffle)
    else:
        fof_pos_all = np.empty(0)
        fof_sh_num_all = np.empty(0)
        count = np.zeros(n_ranks, dtype=int)
        displ = 0
        shuffle = None

    comm.Bcast(count, root=0)

    fof_pos = np.zeros((count[rank], 3), dtype=np.float32)
    fof_sh_num = np.zeros(count[rank], dtype=np.int32)

    comm.Scatterv([fof_pos_all[shuffle], 3*count, 3*displ, MPI.FLOAT], fof_pos)
    comm.Scatterv([fof_sh_num_all[shuffle], count, displ, MPI.INT], fof_sh_num)

    if rank != 0:
        shuffle = np.zeros(sum(count), dtype=int)
    comm.Bcast(shuffle, root=0)

    radii = []
    centers = []
    n_parts = []
    parents = []
    part_ids = np.array([], dtype=int)

    time_start = time.perf_counter()
    for i in range(len(fof_pos)):
        # overall unshuffled index of fof group
        fof_i = shuffle[sum(count[:rank]) + i]
        print(f"{rank} Processing FoF group {fof_i}, {fof_sh_num[i]} subgroups")

        # fof group with no subgroups
        if fof_sh_num[i] == 0:
            print("No subgroups, using fof position", fof_i)
            center_i = fof_pos[i]

            center, radius, n, c_conv, d_conv = find_critical_shell(pd, center_i, crit_dens_a100)

            if c_conv and d_conv and n >= min_n_particles:
                print("Found halo:", center, radius)
                radii.append(radius)
                centers.append(center)
                n_parts.append(n)
                parents.append([fof_i, 0])
                ind = pd.query_radius(center, radius)
                part_ids = np.hstack((part_ids, pd.ids[ind]))
            else:
                print_conv_error(c_conv, d_conv, fof_i)

        # process subgroups
        for j in range(fof_sh_num[i]):
            # overall index of subgroup
            sh_i = sh_offsets[fof_i] + j
            center_i = sh_pos[sh_i]

            # check if initial center is within a sphere already found
            dup = check_duplicate(centers, radii, center_i, check_dup_len=j)
            if dup != -1:
                print_dup_error(fof_i, j, centers[dup], radii[dup])
                continue

            center, radius, n, c_conv, d_conv = find_critical_shell(pd, center_i, crit_dens_a100, findCOM=False)

            # check if final center is within a sphere already found
            dup = check_duplicate(centers, radii, center, radius=radius, check_dup_len=j)
            if dup != -1:
                print_dup_error(fof_i, j, centers[dup], radii[dup])
                continue

            if c_conv and d_conv and n > min_n_particles:
                print("Found halo:", center, radius)
                radii.append(radius)
                centers.append(center)
                n_parts.append(n)
                parents.append([fof_i, sh_i])
                ind = pd.query_radius(center, radius)
                part_ids = np.hstack((part_ids, pd.ids[ind]))
            else:
                print_conv_error(c_conv, d_conv, fof_i, sh_i)

    # TODO: clean up gathering final data
    centers = np.array(centers, dtype=np.float64)
    radii = np.array(radii, dtype=np.float64)
    n_parts = np.array(n_parts)
    parents = np.array(parents)
    len_shells = np.array(len(centers))
    len_particles = np.array(len(part_ids))

    total_length = np.array(0)
    total_particles = np.array(0)
    comm.Reduce(len_shells, total_length, op=MPI.SUM, root=0)
    comm.Reduce(len_particles, total_particles, op=MPI.SUM, root=0)

    lengths_c = np.array(comm.gather(3 * len_shells, root=0))
    lengths_r = np.array(comm.gather(len_shells, root=0))
    lengths_p = np.array(comm.gather(2 * len_shells, root=0))
    lengths_part = np.array(comm.gather(len_particles, root=0))

    displ_c = None
    displ_r = None
    displ_p = None
    displ_part = None
    all_centers = None
    all_radii = None
    all_n_parts = None
    all_parents = None
    all_part_ids = None
    if rank == 0:
        displ_c = np.array([sum(lengths_c[:r]) for r in range(n_ranks)])
        displ_r = np.array([sum(lengths_r[:r]) for r in range(n_ranks)])
        displ_p = np.array([sum(lengths_p[:r]) for r in range(n_ranks)])
        displ_part = np.array([sum(lengths_part[:r]) for r in range(n_ranks)])
        all_centers = np.zeros((total_length, 3))
        all_radii = np.zeros(total_length)
        all_n_parts = np.zeros(total_length, dtype=int)
        all_parents = np.zeros((total_length, 2), dtype=int)
        all_part_ids = np.zeros(total_particles, dtype=int)

    comm.Barrier()
    time_end = time.perf_counter()

    # Gather all data together on node 0
    comm.Gatherv(centers, [all_centers, lengths_c, displ_c, MPI.DOUBLE], root=0)
    comm.Gatherv(radii, [all_radii, lengths_r, displ_r, MPI.DOUBLE], root=0)
    comm.Gatherv(n_parts, [all_n_parts, lengths_r, displ_r, MPI.LONG], root=0)
    comm.Gatherv(parents, [all_parents, lengths_p, displ_p, MPI.LONG], root=0)
    comm.Gatherv(part_ids, [all_part_ids, lengths_part, displ_part, MPI.LONG], root=0)

    if rank == 0:
        masses = pd.part_mass * all_n_parts

        print(f"Finished in {(time_end - time_start)/60:.2f} minutes")
        print(f"{len(all_radii)} critical shells found with more than {min_n_particles}"
              + " particles")

        # save data as hdf5
        if not profile:
            catalog_file = data_dir + "-analysis/critical_shells.hdf5"
            print("Saving filtered catalog to ", catalog_file)
            with h5py.File(catalog_file, 'w') as f:
                # attributes
                f.attrs["Nshells"] = len(all_centers)
                f.attrs["NparticlesTotal"] = len(all_part_ids)
                # datasets
                f.create_dataset("Centers", data=all_centers)
                f.create_dataset("Radii", data=all_radii)
                f.create_dataset("Nparticles", data=all_n_parts)
                f.create_dataset("Masses", data=masses)
                f.create_dataset("Parents", data=all_parents)
                f.create_dataset("ParticleIDs", data=all_part_ids)

        print("Done.")

    if profile:
        pr.disable()
        pr.dump_stats('cpu_%d.prof' % rank)
        with open('cpu_%d.txt' % rank, 'w') as output_file:
            sys.stdout = output_file
            pr.print_stats(sort='time')


if __name__ == "__main__":
    main()
