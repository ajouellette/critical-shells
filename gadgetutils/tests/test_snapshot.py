import os
import numpy as np
import pytest
from .. import snapshot
from .. import utils


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


def test_snapshot_tree():
    pd = snapshot.ParticleData(_data_dir+"/snapshot_002.hdf5",
            load_ids=False, load_vels=False, make_tree=True)
    assert np.isclose(pd.tree.boxsize, pd.box_size).all()
    # test query_radius with center_box_pbc
    inds = pd.query_radius(np.array([0,0,0]), 10)
    pos_centered = utils.center_box_pbc(pd.pos[inds], np.array([0,0,0]), pd.box_size)
    assert (np.linalg.norm(pos_centered, axis=1) <= 10).all()
    exclude = np.delete(pd.pos, inds, axis=0)
    pos_centered = utils.center_box_pbc(exclude, np.array([0,0,0]), pd.box_size)
    assert (np.linalg.norm(pos_centered, axis=1) > 10).all()
    # query_ids should not work when ids are not loaded
    with pytest.raises(RuntimeError):
        pd.query_ids(np.array([1,2,3]))


def test_load_group_data():
    hc = snapshot.HaloCatalog(_data_dir+"/fof_subhalo_tab_002.hdf5")
