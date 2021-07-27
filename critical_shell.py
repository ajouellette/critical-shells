import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import pickle
import h5py
import time
import snapshot

def v_radial(pos, vel, radii=None):
    """Calculate radial velocities wrt origin given cartesian pos and vel."""
    if radii == None:
        radii = np.linalg.norm(pos, axis=1)
    unit_vectors = pos / np.expand_dims(radii, 1)
    return np.multiply(vel, unit_vectors).sum(1)


def vr_physical(cosmo, pos, vel, scale_factor):
    """Calculate physical radial velocities wrt origin.

        comoving positions and GADGET velocities
    """
    radii = np.linalg.norm(pos, axis=1)
    vr_peculiar = np.sqrt(scale_factor) * v_radial(pos, vel, radii=radii)
    hubble_flow = cosmo.H(scale_factor) * scale_factor * radii
    return vr_peculiar + hubble_flow


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
    for i in range(10):
        radii = np.linalg.norm(pos - center, axis=1)
        new_center = np.mean(pos[radii < radius], axis=0)
        #print(new_center, np.linalg.norm(new_center - center))
        if np.linalg.norm(center - new_center) < center_tol:
            center_converged = True
            break
        center = new_center
        radius += 0.05

    # decrease radius and then grow to get critical shell
    radius = 1e-3
    radii = np.linalg.norm(pos - center, axis=1)
    dr = rcrit_tol * 100
    iters = i
    density_converged = False
    while dr > rcrit_tol:
        iters += 1
        new_center = center + np.mean(pos[radii < radius + dr] - center, axis=0)
        new_radii = np.linalg.norm(pos - new_center, axis=1)
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
    pd = snapshot.ParticleData("/projects/caps/aaronjo2/dm-l256-n256-a100/snapshot_021.hdf5")
    #hc = snapshot.HaloCatalog("/home/aaron/dm-l50-n128-a100/fof_tab_012.hdf5")
    fof_tab_data = h5py.File("/projects/caps/aaronjo2/dm-l256-n256-a100/fof_subhalo_tab_021.hdf5", 'r')
    sh_masses = fof_tab_data["Subhalo"]["SubhaloMass"][:] * 1e10
    sh_pos = fof_tab_data["Subhalo"]["SubhaloCM"][:]

    group_offsets = fof_tab_data["Group"]["GroupOffsetType"][:,1]
    group_lens = fof_tab_data["Group"]["GroupLen"][:]
    group_ends = group_offsets + group_lens
    outer_fuzz_start = group_ends[-1]

    sh_group_i = fof_tab_data["Subhalo"]["SubhaloGroupNr"][:]

    fof_tab_data.close()

    print("Particle mass: {:.3e}".format(pd.part_mass))

    # Use H0 = 100 to factor out h
    cosmo = FlatLambdaCDM(H0=100, Om0=pd.OmegaMatter)
    crit_dens_a100 = cosmo.critical_density(-0.99).to(u.Msun / u.Mpc**3).value
    print("Critical density at a = 100: {:.3e}".format(crit_dens_a100))

    radii = []
    centers_s = []

    min_mass = 5e13
    print("{:} subhalos above min mass".format(np.sum(sh_masses > min_mass)))

    check_dup_len = 10

    halos_found = 0
    sh_processed = 0
    time_start = time.perf_counter()
    for i in range(len(sh_masses)):
        if sh_masses[i] < min_mass:
            continue
        
        sh_processed += 1

        if sh_processed % 10 == 0 and sh_processed != 0:
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
        #print("Using {:} positions, {:.1f}% of all".format(len(pos_indicies), 100*len(pos_indicies)/len(pd.pos)))

        center_s, radius, c_converged, d_converged = critical_shell(pd.pos[pos_indicies], center, 
                pd.part_mass, crit_dens_a100)
        if np.sum(center_s < 2) or np.sum(center_s > pd.box_size - 2):
            print("Skipping:", center, "too close to boundary")
            continue

        if c_converged * d_converged:
            # check for duplicate
            for i in range(np.max((0, len(centers_s) - check_dup_len)), len(centers_s)):
                if np.linalg.norm(centers_s[i] - center_s) < radii[i] + radius:
                    print("Skipping duplicate:", centers_s[i], center_s)
                    continue
            print("Found halo:", center, radius)
            radii.append(radius)
            centers_s.append(center_s)
            halos_found += 1
        else:
            print("center converged", c_converged, "density converged", d_converged, radius)


    centers = np.array(centers_s)
    radii = np.array(radii)
    masses = get_mass(radii, crit_density_a100, a=100)
    data = {"centers":centers, "radii":radii, "masses":masses}
    file = open("spherical_halos", "wb")
    pickle.dump(data, file)
    file.close()

