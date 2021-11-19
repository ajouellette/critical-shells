import numpy as np
import numba as nb

nb_parallel = True


@nb.njit(parallel=nb_parallel)
def get_sphere_mask(pos, center, radius):
    """Calculate a spherical mask with given center and radius."""
    return np.sum((pos - center)**2, axis=1) < radius**2


@nb.njit(parallel=nb_parallel)
def mean_pos(pos):
    """Calculate mean position (mean over axis 0)."""
    return np.sum(pos, axis=0) / len(pos)


def wrap_pbc(coords, box_size, in_place=False):
    dx = box_size*(coords < -box_size/2) - box_size*(coords >= box_size/2)
    if in_place:
        coords += dx
        return
    return coords + dx


@nb.njit
def mean_pos_pbc(pos, box_size):
    """Compute center of a cluster in a periodic box.

    Periodic box is assumed to be [0, box_size]
    """
    theta = 2*np.pi * pos / box_size
    x = -mean_pos(np.cos(theta))
    y = -mean_pos(np.sin(theta))
    return box_size * (np.arctan2(y, x) + np.pi) / (2*np.pi)


def sphere_volume(radius, a=1):
    """Volume of a sphere with an optional scale factor."""
    return 4/3 * np.pi * (a * radius)**3
