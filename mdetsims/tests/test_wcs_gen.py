import numpy as np
import galsim

from ..wcs_gen import gen_affine_wcs


def test_gen_affine_wcs():
    seed = 42
    wcs = gen_affine_wcs(
        rng=np.random.RandomState(seed=seed),
        position_angle_range=(0, 360),
        dither_range=(-0.5, 0.5),
        scale=0.25,
        scale_frac_std=0.1,
        shear_std=0.1,
        world_origin=galsim.PositionD(x=20, y=21),
        origin=galsim.PositionD(x=10, y=11))

    rng = np.random.RandomState(seed=seed)
    g1 = rng.normal() * 0.1
    g2 = rng.normal() * 0.1
    scale = (1.0 + rng.normal() * 0.1) * 0.25
    theta = rng.uniform(low=0, high=360)
    dither_u = rng.uniform(
        low=-0.5,
        high=0.5) * scale
    dither_v = rng.uniform(
        low=-0.5,
        high=0.5) * scale

    assert np.allclose(wcs.x0, 10)
    assert np.allclose(wcs.y0, 11)
    assert np.allclose(wcs.u0, 20 + dither_u)
    assert np.allclose(wcs.v0, 21 + dither_v)

    jac_wcs = wcs.jacobian()
    _scale, _shear, _theta, _ = jac_wcs.getDecomposition()

    assert np.allclose(_shear.g1, g1)
    assert np.allclose(_shear.g2, g2)
    assert np.allclose(_scale, scale)
    assert np.allclose(_theta / galsim.degrees, theta)
