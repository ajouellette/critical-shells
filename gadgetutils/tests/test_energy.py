import numpy as np
from .. import energy


def test_potential():
    pos = np.asarray([[0, 0, 0],
                      [1, 1, 1],
                      [0, -1, 0]], dtype=float)
    assert energy.calc_potential(pos, 1, 1) == -(1/np.sqrt(3) + 1 + 1/np.sqrt(6))
