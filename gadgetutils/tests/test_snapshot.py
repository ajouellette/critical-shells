import os
import numpy as np
from .. import snapshot


_test_dir = os.path.dirname(__file__)
_data_dir = os.path.join(_test_dir, "testing_data")


def test_load_particle_data():
    pd = snapshot.ParticleData(_data_dir+"/snapshot_002.hdf5")
    assert pd.n_parts == 64**3
    assert pd.box_size == 50.0
    assert pd.pos.shape == (64**3, 3)
    assert pd.vel.shape == (64**3, 3)
    assert pd.ids.shape == (64**3,)
    assert pd.OmegaLambda == 0.691
    assert pd.OmegaMatter == 0.309
    assert pd.h == 0.677


def test_load_group_data():
    hc = snapshot.HaloCatalog(_data_dir+"/fof_subhalo_tab_002.hdf5")
