import os
import time
import numpy as np
import galsim
import fitsio
import joblib

from ..real_psf import RealPSFGenerator, RealPSF


def test_real_psf_gen_smoke(tmpdir):

    fname = os.path.join(tmpdir.dirname, 'test.fits')
    gen = RealPSFGenerator(
        seed=102,
        scale=0.2143432,
        effective_r0_500=1e5,  # set this really big so we can draw PSFs fast
        im_width=3,
        psf_width=17,
        n_photons=1e5,
        n_lhs=10)

    rng = np.random.RandomState(seed=2398)
    gen.save_to_fits(fname, rng=rng)


def test_real_psf_gen_seed(tmpdir):
    fname1 = os.path.join(tmpdir.dirname, 'test1.fits')
    gen1 = RealPSFGenerator(
        seed=102,
        scale=0.2143432,
        effective_r0_500=1e5,  # set this really big so we can draw PSFs fast
        im_width=3,
        psf_width=17,
        n_photons=1e5,
        n_lhs=10)
    gen1.save_to_fits(fname1)

    fname2 = os.path.join(tmpdir.dirname, 'test2.fits')
    gen2 = RealPSFGenerator(
        seed=102,
        scale=0.2143432,
        effective_r0_500=1e5,  # set this really big so we can draw PSFs fast
        im_width=3,
        psf_width=17,
        n_photons=1e5,
        n_lhs=10)
    gen2.save_to_fits(fname2)

    # files should be eaxctly the same, down to each byte
    with open(fname1, 'rb') as fp1:
        with open(fname2, 'rb') as fp2:
            assert fp1.read() == fp2.read()


def test_real_psf_interp(tmpdir):
    gen = RealPSFGenerator(
        seed=102,
        scale=0.2143432,
        effective_r0_500=1e5,  # set this really big so we can draw PSFs fast
        im_width=3,
        psf_width=17,
        n_photons=1e5,
        n_lhs=10)
    fname = os.path.join(tmpdir.dirname, 'test.fits')
    gen.save_to_fits(fname)

    psf = RealPSF(fname)

    # this test sucks - just making sure it kind of works
    img = gen.get_rec(row=1.9, col=0.5)
    assert np.allclose(
        psf.getPSF(galsim.PositionD(x=0.5, y=1.9)).image.array,
        img,
        atol=2e-3,
        rtol=0)


def test_real_psf_attributes(tmpdir):
    fname = os.path.join(tmpdir.dirname, 'test.fits')
    im_width = 3
    psf_width = 19
    scale = 0.343214
    rng = np.random.RandomState(seed=100)
    images = rng.normal(
        size=(im_width * im_width, psf_width, psf_width)).astype(np.float32)
    psf_locs = rng.uniform(size=(im_width * im_width, 2)) * im_width

    data = np.zeros(1, dtype=[
        ('im_width', 'i8'),
        ('psf_width', 'i8'),
        ('scale', 'f8'),
        ('n_photons', 'f8'),
        ('images', 'f8', images.shape),
        ('locs', 'f8', psf_locs.shape)])
    data['n_photons'] = 1e8
    data['scale'] = scale
    data['locs'][0] = psf_locs
    data['im_width'] = im_width
    data['psf_width'] = psf_width
    data['images'][0] = images

    fitsio.write(fname, data, clobber=True)

    psf = RealPSF(fname)

    assert psf.im_width == im_width
    assert psf.psf_width == psf_width
    assert psf.scale == scale
    assert np.array_equal(psf.psf_images, images)
    assert np.array_equal(psf.psf_locs, psf_locs)


def test_real_psf_pickle(tmpdir):
    gen = RealPSFGenerator(
        seed=102,
        scale=0.2143432,
        effective_r0_500=1e5,  # set this really big so we can draw PSFs fast
        im_width=3,
        psf_width=17,
        n_photons=1e5)

    # first one is slower
    im = gen.getPSF(
        galsim.PositionD(x=1, y=0)
    ).drawImage(
        nx=17, ny=17, scale=0.25, method='phot', n_photons=1e6,
        rng=galsim.BaseDeviate(seed=2))

    t0 = time.time()
    im = gen.getPSF(
        galsim.PositionD(x=1, y=0)
    ).drawImage(
        nx=17, ny=17, scale=0.25, method='phot', n_photons=1e6,
        rng=galsim.BaseDeviate(seed=2))
    t0 = time.time() - t0

    fname = os.path.join(tmpdir.dirname, 'test.pkl')
    joblib.dump(gen, fname)

    genp = joblib.load(fname)

    t0p = time.time()
    imp = genp.getPSF(
        galsim.PositionD(x=1, y=0)
    ).drawImage(
        nx=17, ny=17, scale=0.25, method='phot', n_photons=1e6,
        rng=galsim.BaseDeviate(seed=2))
    t0p = time.time() - t0p

    assert np.array_equal(im.array, imp.array)
    assert np.abs(t0/t0p - 1) < 0.1 or t0p < t0
