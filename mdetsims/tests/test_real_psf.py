import os
import numpy as np
import galsim
import fitsio

from ..real_psf import RealPSF, RealPSFGenerator


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

    psf = RealPSF(fname)

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

    psf = RealPSF(fname)

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
