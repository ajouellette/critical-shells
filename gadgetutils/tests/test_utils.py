import numpy as np
from .. import utils


def test_get_sphere_mask():
    # pick out a sphere from a gaussian cluster
    radius = 0.1
    data = np.random.randn(3000).reshape((1000,3))
    mask = utils.get_sphere_mask(data, np.zeros(3), radius)
    assert (np.linalg.norm(data[mask], axis=1) < radius).all()
    assert (np.linalg.norm(data[np.logical_not(mask)], axis=1) >= radius).all()


def test_sphere_volume():
    assert utils.sphere_volume(1) == 4/3 * np.pi
    assert utils.sphere_volume(1, a=0.5) == 4/3 * np.pi * 0.5**3


def test_mean_pos():
    # mean_pos should give exact same result as np.mean over axis 0
    data = np.random.randn(3000).reshape((1000,3))
    center_np = np.mean(data, axis=0)
    center = utils.mean_pos(data)
    assert np.allclose(center_np, center)


def test_mean_pos_pbc():
    # Make sure mean_pos and mean_pos_pbc give the same
    # results for a cluster not near the boundary
    # use gaussian data centered at (12,12,12) in a box (0,24)
    box = 24
    data = np.random.randn(3000).reshape((1000,3))
    noise_mean = np.mean(data, axis=0)
    data += box/2
    center = box/2 + noise_mean
    center_pbc = utils.mean_pos_pbc(data, box)
    print(center)
    print(center_pbc)
    print(np.linalg.norm(center - center_pbc))
    # For some reason difference between centers is a lot bigger than expected
    # TODO: figure out why and fix, maybe test should be failing?
    assert np.linalg.norm(center - center_pbc) < 0.005

    # gaussian data centered at (10,10,10) in a box (0,11)
    data = np.random.randn(3000).reshape((1000,3))
    noise_mean = np.mean(data, axis=0)
    data += 10
    data[data>11] -= 11
    center = utils.mean_pos_pbc(data, 11)
    print(center)
    print(10+noise_mean)
    print(np.linalg.norm(center - 10 - noise_mean))
    print(np.linalg.norm(center - 10))
    assert np.linalg.norm(center - noise_mean - 10) < 0.005
