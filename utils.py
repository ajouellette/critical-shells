import numpy as np

def vr_physical(pos, vel, Hubble, h=1, a=1, radius=None):
    """Calculate physical radial velocity given simulation pos and vel

    Assumes pos and vel are already centered
    """
    p_radii = np.linalg.norm(pos, axis=1) / h
    if not radius:
        pos_cut 


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
