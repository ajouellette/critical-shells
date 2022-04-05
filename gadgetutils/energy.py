import numpy as np
from .potential import sum_inv_pairdists


def calc_kinetic(vel):
    """Calculate kinetic energy per unit mass given particle velocities."""
    return 0.5 * np.sum(vel * vel, dtype=float)


def calc_potential(pos, mass, G, epsilon=1e-2):
    """Calculate gravitational potential energy per unit mass given
    particle positions."""
    return G * mass * sum_inv_pairdists(pos, epsilon=epsilon)
