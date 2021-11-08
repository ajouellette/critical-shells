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


def center_pbc(data, box_size, center_func="mean"):
    """Compute center of a cluster in a periodic box.

    Periodic box is assumed to be [0, 2pi]
    """
    if center_func == "mean":
        center_func = np.mean
    elif center_func == "median":
        center_func = np.median
    else:
        raise ValueError("unrecognized value for center_func")
    theta = 2*np.pi * data / box_size
    x, y = -func(np.cos(theta), axis=0), -func(np.sin(theta), axis=0)
    return box_size * (np.arctan2(y, x) + np.pi) / (2*np.pi)
