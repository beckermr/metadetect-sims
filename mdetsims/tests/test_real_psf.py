import os
import time
import numpy as np
import galsim
import fitsio
import joblib

from ..real_psf import RealPSFNearest, RealPSFGenerator, RealPSFGP


def test_real_psf_gen(tmpdir):

    fname = os.path.join(tmpdir.dirname, 'test.fits')
    gen = RealPSFGenerator(
        seed=102,
        scale=0.2143432,
        effective_r0_500=1e5,  # set this really big so we can draw PSFs fast
        im_width=3,
        psf_width=17,
        n_photons=1e5)

    rng = np.random.RandomState(seed=2398)
    gen.save_to_fits(fname, rng=rng)

    rng = np.random.RandomState(seed=2398)
    seeds = rng.randint(1, 2**32-1, size=3*3)
    # galsim chokes on np.int64 types
    seed = int(seeds[2*3 + 1])
    gs_rng = galsim.BaseDeviate(seed)

    psf = RealPSFNearest(fname)

    # draw with both and make sure they are the same
    pos = galsim.PositionD(x=1, y=2)
    psf_gen = gen.getPSF(pos)
    psf_gen = psf_gen.drawImage(
        nx=gen.psf_width,
        ny=gen.psf_width,
        scale=gen.scale,
        method='phot',
        n_photons=gen.n_photons,
        rng=gs_rng)
    psf_saved = psf.getPSF(pos)

    assert np.allclose(psf_gen.array, psf_saved.image.array)


def test_real_psf_gen_seed(tmpdir):
    fname1 = os.path.join(tmpdir.dirname, 'test1.fits')
    gen1 = RealPSFGenerator(
        seed=102,
        scale=0.2143432,
        effective_r0_500=1e5,  # set this really big so we can draw PSFs fast
        im_width=3,
        psf_width=17,
        n_photons=1e5)
    gen1.save_to_fits(fname1)

    fname2 = os.path.join(tmpdir.dirname, 'test2.fits')
    gen2 = RealPSFGenerator(
        seed=102,
        scale=0.2143432,
        effective_r0_500=1e5,  # set this really big so we can draw PSFs fast
        im_width=3,
        psf_width=17,
        n_photons=1e5)
    gen2.save_to_fits(fname2)

    # files should be eaxctly the same, down to each byte
    with open(fname1, 'rb') as fp1:
        with open(fname2, 'rb') as fp2:
            assert fp1.read() == fp2.read()


def test_real_psf(tmpdir):
    im_width = 3
    psf_width = 19
    scale = 0.343214
    rng = np.random.RandomState(seed=100)
    images = rng.normal(
        size=(im_width, im_width, psf_width, psf_width)).astype(np.float32)
    fname = os.path.join(tmpdir.dirname, 'test.fits')
    d = np.zeros(1, dtype=[
        ('im_width', 'i8'),
        ('psf_width', 'i8'),
        ('scale', 'f8'),
        ('flat_image', 'f4', im_width**2 * psf_width**2)])
    d['im_width'] = im_width
    d['psf_width'] = psf_width
    d['scale'] = scale
    d['flat_image'][0] = images.flatten()

    fitsio.write(fname, d, clobber=True)

    psf = RealPSFNearest(fname)

    assert psf.im_width == im_width
    assert psf.psf_width == psf_width
    assert psf.scale == scale
    assert np.array_equal(psf.psf_images, images)

    assert np.allclose(
        psf.getPSF(galsim.PositionD(x=0.5, y=1.9)).image.array,
        images[1, 0])

    assert np.allclose(
        psf.getPSF(galsim.PositionD(x=-0.5, y=-1.9)).image.array,
        images[0, 0])

    assert np.allclose(
        psf.getPSF(galsim.PositionD(x=100.5, y=100.9)).image.array,
        images[2, 2])


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


def test_real_psf_gen_grid(tmpdir):
    fname = os.path.join(tmpdir.dirname, 'test.fits')
    gen = RealPSFGenerator(
        seed=102,
        scale=0.2143432,
        effective_r0_500=1e5,  # set this really big so we can draw PSFs fast
        im_width=25,
        grid_spacing=5,
        psf_width=17,
        n_photons=1e5)

    rng = np.random.RandomState(seed=2398)
    gen.save_to_fits(fname, rng=rng)

    d = fitsio.read(fname)
    assert d['im_width'][0] == 25
    assert d['psf_width'][0] == 17
    assert d['grid_spacing'][0] == 5
    assert d['flat_image'][0].shape[0] == 6*6*17*17

    rng = np.random.RandomState(seed=2398)
    seeds = rng.randint(1, 2**32-1, size=5*5)
    # galsim chokes on np.int64 types
    seed = int(seeds[2*3 + 1])
    gs_rng = galsim.BaseDeviate(seed)

    psf = RealPSFGP(fname)

    # draw with both and make sure they are the same
    pos = galsim.PositionD(x=1, y=2)
    psf_saved = psf.getPSF(pos)
    psf_gen = gen.getPSF(galsim.PositionD(x=1, y=2))
    psf_gen = psf_gen.drawImage(
        nx=gen.psf_width,
        ny=gen.psf_width,
        scale=gen.scale,
        method='phot',
        n_photons=gen.n_photons,
        rng=gs_rng)

    # not a great test, but makes sure it doesn't get worse
    assert np.median(np.abs(psf_gen.array - psf_saved.image.array)) < 2e-5
