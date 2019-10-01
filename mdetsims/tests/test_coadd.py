import numpy as np
import galsim

import pytest

from ..coadd import (
    invert_affine_transform_wcs,
    coadd_psfs,
    coadd_image_noise_interpfrac)
from ..wcs_gen import gen_affine_wcs


def test_invert_affine_transform_wcs():
    wcs = galsim.AffineTransform(
        dudx=0.5,
        dudy=-3.0,
        dvdx=0.7,
        dvdy=-10,
        origin=galsim.PositionD(x=9.3, y=-11),
        world_origin=galsim.PositionD(x=-900, y=100),
    )
    rng = np.random.RandomState(seed=10)
    u = rng.uniform(low=-100, high=100, size=10)
    v = rng.uniform(low=-100, high=100, size=10)

    x, y = invert_affine_transform_wcs(u, v, wcs)

    for i in range(10):
        world_pos = galsim.PositionD(x=u[i], y=v[i])
        pos = wcs.toImage(world_pos)
        assert np.allclose(pos.x, x[i])
        assert np.allclose(pos.y, y[i])


@pytest.mark.parametrize('crazy_psf', [False, True])
@pytest.mark.parametrize('crazy_wcs', [False, True])
def test_coadd_psfs(crazy_wcs, crazy_psf):
    n_coadd = 10
    rng = np.random.RandomState(seed=42)

    if crazy_psf:
        psfs = []
        for _ in range(n_coadd):
            _fwhm = 0.9 * (1.0 + rng.normal() * 0.1)
            _g1 = rng.normal() * 0.3
            _g2 = rng.normal() * 0.3
            psfs.append(galsim.Gaussian(fwhm=_fwhm).shear(g1=_g1, g2=_g2))
    else:
        psfs = []
        for _ in range(n_coadd):
            psfs.append(galsim.Gaussian(fwhm=0.9).shear(g1=-0.1, g2=0.3))

    coadd_dim = 225
    coadd_cen = (coadd_dim - 1)/2

    se_dim = int(np.ceil(coadd_dim * np.sqrt(2)))
    if se_dim % 2 == 0:
        se_dim += 1
    se_cen = (se_dim - 1)/2
    scale = 0.27

    psf_dim = 33
    psf_dim_half = (psf_dim - 1)/2

    se_psf_dim = int(np.ceil(psf_dim * np.sqrt(2)))
    if se_psf_dim % 2 == 0:
        se_psf_dim += 1
    se_psf_dim_half = (se_psf_dim - 1)/2

    coadd_offset = coadd_cen - psf_dim_half

    world_origin = galsim.PositionD(
        x=coadd_cen*scale, y=coadd_cen*scale)
    origin = galsim.PositionD(x=se_cen, y=se_cen)

    coadd_wgts = rng.uniform(size=n_coadd)
    coadd_wgts /= np.sum(coadd_wgts)

    se_wcs_objs = []

    for _ in range(n_coadd):
        if crazy_wcs:
            wcs = gen_affine_wcs(
                rng=rng,
                position_angle_range=(0, 360),
                dither_range=(5, 5),
                scale=scale,
                scale_frac_std=0.05,
                shear_std=0.1,
                world_origin=world_origin,
                origin=origin)
        else:
            wcs = gen_affine_wcs(
                rng=rng,
                position_angle_range=(0, 0),
                dither_range=(0, 0),
                scale=scale,
                scale_frac_std=0.0,
                shear_std=0.0,
                world_origin=world_origin,
                origin=origin)

        se_wcs_objs.append(wcs)

    se_psfs = []
    se_offsets = []
    for wcs, psf in zip(se_wcs_objs, psfs):
        pos = wcs.toImage(world_origin)
        xp = int(pos.x + 0.5)
        yp = int(pos.y + 0.5)
        dx = pos.x - xp
        dy = pos.y - yp
        se_offsets.append((xp - se_psf_dim_half, yp - se_psf_dim_half))
        se_psfs.append(psf.drawImage(
            nx=se_psf_dim,
            ny=se_psf_dim,
            wcs=wcs.local(world_pos=world_origin),
            offset=galsim.PositionD(x=dx, y=dy)).array)

    coadd_psf = coadd_psfs(
        se_psfs, se_wcs_objs, coadd_wgts,
        scale, psf_dim, coadd_offset, se_offsets)

    true_coadd_psf = galsim.Sum(
        [psf.withFlux(wgt) for psf, wgt in zip(psfs, coadd_wgts)]
    ).drawImage(
        nx=psf_dim,
        ny=psf_dim,
        scale=scale).array
    true_coadd_psf /= np.sum(true_coadd_psf)

    # IDK if the code is right, but it works to machine precision when
    # everything is aligned
    if not crazy_psf and not crazy_wcs:
        rtol = 0
        atol = 1e-7
    else:
        rtol = 0
        atol = 8e-4

    if not np.allclose(coadd_psf, true_coadd_psf, rtol=rtol, atol=atol):
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, axs = plt.subplots(nrows=2, ncols=2)
        sns.heatmap(true_coadd_psf, ax=axs[0, 0])
        sns.heatmap(coadd_psf, ax=axs[0, 1])
        sns.heatmap(coadd_psf - true_coadd_psf, ax=axs[1, 0])
        sns.heatmap(
            np.abs(coadd_psf - true_coadd_psf) <
            atol + rtol * np.abs(true_coadd_psf),
            ax=axs[1, 1])
        print(np.max(np.abs(coadd_psf - true_coadd_psf)))

    assert np.allclose(coadd_psf, true_coadd_psf, rtol=rtol, atol=atol)


@pytest.mark.parametrize('crazy_obj', [False, True])
@pytest.mark.parametrize('crazy_wcs', [False, True])
def test_coadd_image_noise_interpfrac(crazy_wcs, crazy_obj):
    n_coadd = 10
    rng = np.random.RandomState(seed=42)

    if crazy_obj:
        objs = []
        for _ in range(n_coadd):
            _fwhm = 0.9 * (1.0 + rng.normal() * 0.1)
            _g1 = rng.normal() * 0.3
            _g2 = rng.normal() * 0.3
            objs.append(galsim.Gaussian(fwhm=_fwhm).shear(g1=_g1, g2=_g2))
    else:
        objs = []
        for _ in range(n_coadd):
            objs.append(galsim.Gaussian(fwhm=0.9).shear(g1=-0.1, g2=0.3))

    coadd_dim = 21
    coadd_cen = (coadd_dim - 1)/2

    se_dim = int(np.ceil(coadd_dim * np.sqrt(2)))
    if se_dim % 2 == 0:
        se_dim += 1
    se_cen = (se_dim - 1)/2
    scale = 0.27

    world_origin = galsim.PositionD(
        x=coadd_cen*scale, y=coadd_cen*scale)
    origin = galsim.PositionD(x=se_cen, y=se_cen)

    coadd_wgts = rng.uniform(size=n_coadd)
    coadd_wgts /= np.sum(coadd_wgts)

    se_wcs_objs = []

    for _ in range(n_coadd):
        if crazy_wcs:
            wcs = gen_affine_wcs(
                rng=rng,
                position_angle_range=(0, 360),
                dither_range=(-5, 5),
                scale=scale,
                scale_frac_std=0.05,
                shear_std=0.1,
                world_origin=world_origin,
                origin=origin)
        else:
            wcs = gen_affine_wcs(
                rng=rng,
                position_angle_range=(0, 0),
                dither_range=(0, 0),
                scale=scale,
                scale_frac_std=0.0,
                shear_std=0.0,
                world_origin=world_origin,
                origin=origin)

        se_wcs_objs.append(wcs)

    se_images = []
    se_noises = []
    se_interp_fracs = []
    for wcs, obj in zip(se_wcs_objs, objs):
        pos = wcs.toImage(world_origin)
        dx = pos.x - se_cen
        dy = pos.y - se_cen
        se_images.append(obj.drawImage(
            nx=se_dim,
            ny=se_dim,
            wcs=wcs.local(world_pos=world_origin),
            offset=galsim.PositionD(x=dx, y=dy)).array)
        se_noises.append(rng.normal(size=(se_dim, se_dim)))
        se_interp_fracs.append(rng.uniform(size=(se_dim, se_dim)))

    coadd_img, coadd_nse, coadd_intp = coadd_image_noise_interpfrac(
        se_images, se_noises, se_interp_fracs, se_wcs_objs,
        coadd_wgts, scale, coadd_dim)

    true_coadd_img = galsim.Sum(
        [obj.withFlux(wgt) for obj, wgt in zip(objs, coadd_wgts)]
    ).drawImage(
        nx=coadd_dim,
        ny=coadd_dim,
        scale=scale).array

    if not crazy_obj and not crazy_wcs:
        rtol = 0
        atol = 1e-7
    else:
        rtol = 0
        atol = 8e-4

    if not np.allclose(coadd_img, true_coadd_img, rtol=rtol, atol=atol):
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, axs = plt.subplots(nrows=2, ncols=2)
        sns.heatmap(true_coadd_img, ax=axs[0, 0])
        sns.heatmap(coadd_img, ax=axs[0, 1])
        sns.heatmap(coadd_img - true_coadd_img, ax=axs[1, 0])
        sns.heatmap(
            np.abs(coadd_img - true_coadd_img) <
            atol + rtol * np.abs(true_coadd_img),
            ax=axs[1, 1])
        print(np.max(np.abs(coadd_img - true_coadd_img)))

    assert np.allclose(coadd_img, true_coadd_img, rtol=rtol, atol=atol)
