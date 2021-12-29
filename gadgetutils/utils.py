import numpy as np
import numba as nb

nb_parallel = True  # not sure if this makes any difference on the cluster


@nb.njit(parallel=nb_parallel)
def get_sphere_mask(pos, center, radius):
    """Calculate a spherical mask with given center and radius.

    Returns a boolean mask that filters out the particles given by pos
    that are within the given radius from center.
    """
    return np.sum((pos - center)**2, axis=1) < radius**2


@nb.njit(parallel=nb_parallel)
def mean_pos(pos):
    """Calculate mean position (mean over axis 0).

    Needed since numba does not support np.mean with axis argument.
    Warning: numba does not support the dtype argument, so numeric errors
    may occur if data is single precision.
    """
    return np.sum(pos, axis=0) / len(pos)


@nb.njit
def mean_pos_pbc(pos, box_size):
    """Compute center of a cluster in a periodic box.

    Periodic box is assumed to be [0, box_size]

    Method from Bai and Breen 2008.
    (https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions)
    """
    theta = 2*np.pi * pos / box_size
    x = -mean_pos(np.cos(theta))
    y = -mean_pos(np.sin(theta))
    return box_size * (np.arctan2(y, x) + np.pi) / (2*np.pi)


def wrap_pbc(coords, box_size, in_place=False):
    dx = box_size*(coords < -box_size/2) - box_size*(coords >= box_size/2)
    if in_place:
        coords += dx
        return
    return coords + dx


@nb.njit
def center_box_pbc(coords, center, box_size):
    """Recenter positions in a periodic box on a given center.

    Returns coordinates in range [-L/2, L/2) centered on given center.
    """
    dx = center + box_size*(center < -box_size/2) - box_size*(center >= box_size/2)
    coords_centered = coords - dx
    coords_centered += box_size*(coords_centered < -box_size/2) \
                    - box_size*(coords_centered >= box_size/2)
    return coords_centered


def sphere_volume(radius, a=1):
    """Volume of a sphere with an optional scale factor."""
    return 4/3 * np.pi * (a * radius)**3


@nb.njit
def calc_vr_phys(pos, vel, center, radius, a, H, h):
    """Calculate physical radial velocities.

    Calculates physical radial velocities (km/s) wrt to given center for all
    particles given by (pos, vel).
    (center, radius) determines the boundary of the halo which is used to calculate
    the velocity of the center of mass.
    H is the Hubble parameter at the given scale factor a,
    while h = H0/100.
    Returns distances of particles from center as well as radial velocities.
    """
    pos_centered = pos - center
    p_radii = np.sqrt(np.sum(pos_centered**2, axis=1))
    vel_center = mean_pos(vel[p_radii < radius])
    vel_peculiar = np.sqrt(a) * (vel - vel_center)
    unit_vectors = pos_centered / np.expand_dims(p_radii, 1)
    vr_peculiar = np.multiply(vel_peculiar, unit_vectors).sum(1)
    hubble_flow = H * a * p_radii / h
    vr_physical = vr_peculiar + hubble_flow
    return p_radii, vr_physical


@nb.njit
def calc_hmf(bins, masses, box_size):
    """Calculate the halo mass function given masses and bins.

    Returns number densitesy of halos more massive than given bin values and
    errors assuming a Poisson counting process.
    """
    hmf = np.zeros_like(bins)
    errors = np.zeros_like(bins)
    for i, thresh in enumerate(bins):
        count = np.sum(masses > thresh)
        hmf[i] = count / box_size**3
        errors[i] = np.sqrt(count) / box_size**3
    return hmf, errors
