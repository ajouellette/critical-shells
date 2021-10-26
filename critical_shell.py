import numpy as np
from mpi4py import MPI
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import pickle
import h5py
import sys
import os
import snapshot


def get_density(radii, cut_radius, part_mass, a=1):
    """Calculate density of particles enclosed by a radius.

    Parameters:
    radii: ndarray (n,) - Array of all particle radii
    cut_radius: float - Radius of shell
    part_mass: float - Particle mass
    a: float, optional - Scale factor

    Returns:
    density: float
    """
    num_particles = np.sum(radii < cut_radius)
    volume = 4/3 * np.pi * (cut_radius * a)**3
    return num_particles * part_mass / volume


def get_mass(radius, density, a=1):
    """Calculate mass enclosed by a spherical shell.

    Parameters:
    radius: float or ndarray - Radius of shell
    density: float or ndarray - Density of shell
    a: float, optional - Scale factor

    Returns:
    mass: float or ndarray
    """
    volume = 4/3 * np.pi * (radius * a)**3
    return density * volume


def critical_shell(pos, center, part_mass, crit_dens, crit_ratio=2, center_tol=1e-3, rcrit_tol=5e-4, maxiters=100):
    """Find a critical shell starting from given center.

    Parameters:
    pos: ndarray (n,3) - Positions of all particles
    center: ndarray (1,3) - Initial guess for center of shell
    part_mass: float - Particle mass
    crit_dens: float - Critical density of universe
    crit_ratio: float - Ratio of critical shell density to critical density of universe
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
    radius = 5e-3
    # find center of largest substructure
    center_converged = False
    for i in range(20):
        radii = np.sqrt(np.sum((pos - center)**2, axis=1))
        new_center = np.mean(pos[radii < radius], axis=0)
        if np.linalg.norm(center - new_center) < center_tol:
            center_converged = True
            break
        center = new_center
        radius += 5e-3

    # decrease radius and then grow to get critical shell
    radius = 5e-4
    radii = np.sqrt(np.sum((pos - center)**2, axis=1))
    dr = rcrit_tol * 100
    iters = i
    density_converged = False
    while dr > rcrit_tol:
        iters += 1
        if np.sum(radii < radius+dr) < 2:
            dr *= 2
        new_center = center + np.mean(pos[radii < radius + dr] - center, axis=0)
        new_radii = np.sqrt(np.sum((pos - new_center)**2, axis=1))
        density = get_density(new_radii, radius+dr, part_mass, a=100)

        if density < crit_ratio * crit_dens and density > 0:
            dr /= 2
        else:
            center = new_center
            radii = new_radii
            radius += dr
        if iters > maxiters:
            break

    # number of particles
    n_parts = np.sum(new_radii < radius)

    # convergence check (maybe not necessary)
    if density < crit_ratio * crit_dens and density > crit_dens and dr < rcrit_tol:
        density_converged = True

    return center, radius, n_parts, center_converged, density_converged


def print_conv_error(center_conv, density_conv, fof_i):
    print("Did not converge {}:".format(fof_i), "center" if not c_conv else '',
            "density" if not d_conv else '')


def print_dup_error(fof_i, sh_i, center, radius):
    print("Skipping duplicate: fof {}, sh {} ".format(fof_i, sh_i), center, radius)


def check_duplicate(centers, radii, center, radius=0, check_dup_len=10):
    """Check if a shell is a duplicate or contained within another shell."""
    for i in range(np.max((0, len(centers) - check_dup_len)), len(centers)):
        if np.linalg.norm(centers[i] - center) < radii[i] + radius:
            return i
    return -1


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    np.random.seed(10)

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

    with h5py.File(snap_file) as f:
        box_size = f["Header"].attrs["BoxSize"]
        Om0 = f["Parameters"].attrs["Omega0"]
        part_mass = f["Header"].attrs["MassTable"][1] * 1e10
        pos = f["PartType1"]["Coordinates"][:]

    # Use H0 = 100 to factor out h, calculate critical density
    cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
    crit_dens_a100 = cosmo.critical_density(-0.99).to(u.Msun / u.Mpc**3).value

    with h5py.File(fof_file) as fof_tab_data:
        if rank == 0:
            fof_pos_all = fof_tab_data["Group"]["GroupPos"][:]
            fof_sh_num_all = fof_tab_data["Group"]["GroupNsubs"][:]

        comm.Barrier()

        sh_pos = fof_tab_data["Subhalo"]["SubhaloPos"][:]
        sh_offsets = fof_tab_data["Group"]["GroupFirstSub"][:]

    if rank == 0:
        # some debug info
        print("Critical density at a = 100: {:.3e}".format(crit_dens_a100))
        print("Particle mass: {:.3e}".format(part_mass))

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

    fof_pos = np.zeros((count[rank],3), dtype=np.float32)
    fof_sh_num = np.zeros(count[rank], dtype=np.int32)

    comm.Scatterv([fof_pos_all[shuffle], 3*count, 3*displ, MPI.FLOAT], fof_pos)
    comm.Scatterv([fof_sh_num_all[shuffle], count, displ, MPI.INT], fof_sh_num)

    if rank != 0:
        shuffle = np.zeros(sum(count), dtype=int)
    comm.Bcast(shuffle, root=0)

    radii = []
    centers = []
    n_parts = []

    for i in range(len(fof_pos)):

        # overall unshuffled index of fof group
        fof_i = shuffle[sum(count[:rank]) + i]
        print("{} Processing FoF group {}, {} subgroups".format(rank, fof_i, fof_sh_num[i]))

        # only use subset of all positions to speed up critical shell search
        # cube 4 Mpc across, centered on FoF group
        cubic_mask = np.product(np.abs(pos - fof_pos[i]) < 2, axis=1, dtype=bool)
        pos_cut = pos[cubic_mask]

        # fof group with no subgroups
        if fof_sh_num[i] == 0:
            print("No subgroups, using fof position", fof_i)
            center_i = fof_pos[i]

            # ignore centers too close to box boundary
            if np.sum(center_i < 2) or np.sum(center_i > box_size - 2):
                print("Skipping:", center_i, "too close to boundary")
                continue

            center, radius, n, c_conv, d_conv = critical_shell(pos_cut, center_i,
                    part_mass, crit_dens_a100)

            if c_conv and d_conv:
                print("Found halo:", center, radius)
                radii.append(radius)
                centers.append(center)
                n_parts.append(n)
            else:
                print_conv_error(c_conv, d_conv, fof_i)

        # process subgroups
        for j in range(fof_sh_num[i]):
            # overall index of subgroup
            sh_i = sh_offsets[fof_i] + j
            center_i = sh_pos[sh_i]

            # ignore centers too close to box boundary
            if np.sum(center_i < 2) or np.sum(center_i > box_size - 2):
                print("Skipping:", center_i, "too close to boundary")
                continue

            # check if initial center is within a sphere already found
            dup = check_duplicate(centers, radii, center_i, check_dup_len=j)
            if dup != -1:
                print_dup_error(fof_i, j, centers[dup], radii[dup])
                continue

            center, radius, n, c_conv, d_conv = critical_shell(pos_cut, center_i,
                    part_mass, crit_dens_a100)

            # check if final center is within a sphere already found
            dup = check_duplicate(centers, radii, center, radius=radius, check_dup_len=j)
            if dup != -1:
                print_dup_error(fof_i, j, centers[dup], radii[dup])
                continue

            if c_conv and d_conv:
                print("Found halo:", center, radius)
                radii.append(radius)
                centers.append(center)
                n_parts.append(n)
            else:
                print_conv_error(c_conv, d_conv, fof_i)


    # TODO: clean up gathering final data
    centers = np.array(centers, dtype=np.float64)
    radii = np.array(radii, dtype=np.float64)
    n_parts = np.array(n_parts)
    length = np.array(len(centers))

    total_length = np.array(0)
    comm.Reduce(length, total_length, op=MPI.SUM, root=0)

    lengths_c = np.array(comm.gather(3 * length, root=0))
    lengths_r = np.array(comm.gather(length, root=0))

    displ_c = None
    displ_r = None
    all_centers = None
    all_radii = None
    all_n_parts = None
    if rank == 0:
        displ_c = np.array([sum(lengths_c[:r]) for r in range(n_ranks)])
        displ_r = np.array([sum(lengths_r[:r]) for r in range(n_ranks)])
        all_centers = np.zeros((total_length,3))
        all_radii = np.zeros(total_length)
        all_n_parts = np.zeros(total_length, dtype=int)

    comm.Barrier()

    # Gather all data together on node 0
    comm.Gatherv(centers, [all_centers, lengths_c, displ_c, MPI.DOUBLE], root=0)
    comm.Gatherv(radii, [all_radii, lengths_r, displ_r, MPI.DOUBLE], root=0)
    comm.Gatherv(n_parts, [all_n_parts, lengths_r, displ_r, MPI.LONG], root=0)

    if rank == 0:
        filter_n = True
        print("Finished, {} spherical halos found".format(len(all_radii)))
        min_n = 10
        n_cut = all_n_parts >= 10
        print("{} spherical halos found with more than {} particles".format(np.sum(n_cut), min_n))
        if filter_n:
            print("Saving filtered catalog")
            all_centers = all_centers[n_cut]
            all_radii = all_radii[n_cut]
            all_n_parts = all_n_parts[n_cut]

        masses = part_mass * all_n_parts
        data = {"centers":all_centers, "radii":all_radii, "masses":masses}

        with open(data_dir + "-analysis/spherical_halos_mpi", "wb") as f:
            pickle.dump(data, f)
