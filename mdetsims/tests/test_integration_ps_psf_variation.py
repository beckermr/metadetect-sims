import numpy as np
import galsim

import pytest

from ..sim_utils import Sim

PIXEL_SCALE = 0.263


class GridPSPSFSim(Sim):
    """A sim, but the sources are on a grid and they are point sources"""
    def __init__(self, *, n_coadd, n_coadd_psf, homogenize_psf):
        super().__init__(
            rng=np.random.RandomState(seed=10),
            gal_type='exp',
            psf_type='ps',
            scale=PIXEL_SCALE,
            shear_scene=True,
            n_coadd=n_coadd,
            n_coadd_psf=n_coadd_psf,
            g1=0.0,
            g2=0.0,
            dim=225,
            buff=25,
            noise=8.0,
            ngal=25/((225-25*2) * PIXEL_SCALE/60)**2,
            psf_kws=None,
            homogenize_psf=homogenize_psf)

    def _get_dxdy(self):
        if not hasattr(self, '_pos_grid'):
            self._pos_grid = []
            for x in np.linspace(-self.pos_width, self.pos_width, 5):
                for y in np.linspace(-self.pos_width, self.pos_width, 5):
                    self._pos_grid.append((x, y))
            self._pos_grid_ind = -1
        self._pos_grid_ind += 1
        return self._pos_grid[self._pos_grid_ind]

    def _get_gal_exp(self):
        flux = 10**(0.4 * (30 - 20))
        half_light_radius = 1e-5

        obj = galsim.Sersic(
            half_light_radius=half_light_radius,
            n=1,
        ).withFlux(flux)

        return [obj]

    def _get_nobj(self):
        return self.nobj


def _get_fwhm_g1g2(im):
    mom = galsim.hsm.FindAdaptiveMom(im)
    return (
        mom.moments_sigma * im.scale,
        mom.observed_shape.g1, mom.observed_shape.g2)


def test_integration_ps_psf_variation_params():
    s = GridPSPSFSim(n_coadd=100, n_coadd_psf=1, homogenize_psf=False)
    s.get_mbobs()
    assert len(s._psfs) == 1
    assert s.nobj == 25
    assert not s.homogenize_psf


@pytest.mark.parametrize(
    'n_coadd,n_coadd_psf,homogenize_psf,should_vary',
    [(100, 1, False, True),
     (100, 100, False, False),
     (100, 1, True, False),
     ])
def test_integration_ps_psf_variation_stamps_var(
        n_coadd, n_coadd_psf, homogenize_psf, should_vary):
    s = GridPSPSFSim(
        n_coadd=n_coadd,
        n_coadd_psf=n_coadd_psf,
        homogenize_psf=homogenize_psf)
    mbobs = s.get_mbobs()

    # extract stamps, measure variation in output objects
    centers = (
        np.linspace(-s.pos_width, s.pos_width, 5) / s.scale + s.im_cen)
    ssize = 17
    scen = (ssize - 1) / 2
    data = []
    for xc in centers:
        for yc in centers:
            xll = int(xc - scen + 1)
            yll = int(yc - scen + 1)
            stamp = galsim.ImageD(
                mbobs[0][0].image[yll:yll+ssize, xll:xll+ssize].copy(),
                scale=s.scale)
            data.append(_get_fwhm_g1g2(stamp))

    fwhms, g1, g2 = zip(*data)

    if should_vary:
        assert np.std(fwhms)/np.mean(fwhms) > 0.02
        assert np.max(g1) - np.min(g1) > 0.02
        assert np.max(g2) - np.min(g2) > 0.02
    else:
        assert np.std(fwhms)/np.mean(fwhms) < 0.02
        assert np.max(g1) - np.min(g1) < 0.02
        assert np.max(g2) - np.min(g2) < 0.02

    print(
        np.std(fwhms)/np.mean(fwhms),
        np.max(g1) - np.min(g1),
        np.max(g2) - np.min(g2))
