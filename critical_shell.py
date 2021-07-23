import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import pickle
import h5py
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

def critical_shell(pos, center, part_mass, crit_dens, crit_ratio=2, center_tol=1e-3, rcrit_tol=5e-4):
    """Find a critical shell starting from given center."""
    radius = 0.1
    print(center)
    # find center of largest structure
    center_converged = False
    for i in range(10):
        radii = np.linalg.norm(pos - center, axis=1)
        new_center = np.mean(pos[radii < radius], axis=0)
        print(new_center, np.linalg.norm(new_center - center))
        if np.linalg.norm(center - new_center) < center_tol:
            center_converged = True
            break
        center = new_center
        radius += 0.05

    # decrease radius to get critical shell
    radius = 4e-2
    radii = np.linalg.norm(pos - center, axis=1)
    density_converged = False
    # decide whether to grow or shrink sphere
    if get_density(radii, radius, part_mass, a=100) > crit_ratio*crit_dens:
        dr = rcrit_tol
    else:
        dr = -rcrit_tol
    for i in range(100):
        center += np.mean(pos[radii < radius] - center, axis=0)
        #print(center, radius)
        radii = np.linalg.norm(pos - center, axis=1)
        density = get_density(radii, radius, part_mass, a=100)
        #print("{:.3f}".format(density / crit_dens))
        if density <= crit_ratio * crit_dens and density > crit_dens:
            density_converged = True
            break
        radius += dr

    return center, radius, center_converged*density_converged


def crit_shell_from_subhalos(fof_tab, min_mass=1e12):
    fof_tab_data = h5py.File(fof_tab)
    sh_masses = fof_tab_data["Subhalo"]["SubhaloMass"]
    sh_pos = fof_tab_data["Subhalo"]["SubhaloCM"]

    for i in range(len(sh_masses)):
        if sh_masses * 1e10 < min_mass:
            continue

        # ignore centers too close to box boundary
        if np.sum(center < 2) or np.sum(center > pd.box_size - 2):
            print()
            print("Skipping ", center)
            print()
            continue

        critical_shell(pos, sh_pos[i], )

    return


if __name__ == "__main__":
    pd = snapshot.ParticleData("/home/aaron/dm-l50-n128-a100/snapshot_012.hdf5")
    hc = snapshot.HaloCatalog("/home/aaron/dm-l50-n128-a100/fof_tab_012.hdf5")

    # Use H0 = 100 to factor out h
    cosmo = FlatLambdaCDM(H0=100, Om0=pd.OmegaMatter)
    crit_dens_a100 = cosmo.critical_density(-0.99).to(u.Msun / u.Mpc**3).value

    # use x most massive halos
    centers = hc.pos[:300]
    radii = []
    centers_s = []

    for center in centers:
        # ignore centers too close to box boundary
        if np.sum(center < 2) or np.sum(center > pd.box_size - 2):
            print()
            print("Skipping ", center)
            print()
            continue

        center_s, radius, converged = critical_shell(pd.pos, center, pd.part_mass, crit_dens_a100)
        if np.sum(center_s < 2) or np.sum(center_s > pd.box_size - 2):
            print()
            print("Skipping ", center)
            print()
            continue

        if converged:
            radii.append(radius)
            centers_s.append(center_s)

        #print(critical_shell(pd.pos, center, pd.part_mass, crit_dens_a100))

    data = {"centers":np.array(centers), "radii":np.array(radii)}
    file = open("spherical_halos", "wb")
    pickle.dump(data, file)
    file.close()

