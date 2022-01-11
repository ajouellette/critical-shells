import numpy as np
from .. import utils


def test_sphere_volume():
    assert utils.sphere_volume(1) == 4/3 * np.pi
    assert utils.sphere_volume(1, a=0.5) == 4/3 * np.pi * 0.5**3


def test_mean_pos_pbc():
    # Make sure mean_pos and mean_pos_pbc give the same
    # results for a cluster not near the boundary
    # use gaussian data centered at (12,12,12) in a box (0,24)
    box = 25
    data = box/2 + np.random.randn(3000).reshape((1000,3))
    print(utils.mean_pos(data))
    print(utils.mean_pos_pbc(data, box))
    assert np.allclose(utils.mean_pos(data), utils.mean_pos_pbc(data, box))
