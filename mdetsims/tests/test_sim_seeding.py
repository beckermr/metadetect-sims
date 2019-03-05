import numpy as np

import pytest

from ..sim_utils import Sim


@pytest.mark.parametrize('gal_type', ['exp'])
@pytest.mark.parametrize('psf_type', ['gauss', 'ps'])
@pytest.mark.parametrize('homogenize_psf', [False, True])
@pytest.mark.parametrize('n_coadd_psf', [1, 2, 3])
def test_sim_seeding_reproduce(
        gal_type, psf_type, homogenize_psf, n_coadd_psf):

    s1 = Sim(
        rng=np.random.RandomState(seed=10),
        gal_type=gal_type,
        psf_type=psf_type,
        homogenize_psf=homogenize_psf,
        n_coadd_psf=n_coadd_psf,
        n_coadd=10)

    s2 = Sim(
        rng=np.random.RandomState(seed=10),
        gal_type=gal_type,
        psf_type=psf_type,
        homogenize_psf=homogenize_psf,
        n_coadd_psf=n_coadd_psf,
        n_coadd=10)

    mbobs1 = s1.get_mbobs()
    mbobs2 = s2.get_mbobs()

    assert np.array_equal(mbobs1[0][0].image, mbobs2[0][0].image)
    assert np.array_equal(mbobs1[0][0].noise, mbobs2[0][0].noise)
    assert np.array_equal(mbobs1[0][0].psf.image, mbobs2[0][0].psf.image)


@pytest.mark.parametrize('gal_type', ['exp'])
@pytest.mark.parametrize('psf_type', ['ps', 'gauss'])
@pytest.mark.parametrize('homogenize_psf', [False, True])
@pytest.mark.parametrize('n_coadd_psf', [1, 2, 3])
def test_sim_seeding_not_reproduce(
        gal_type, psf_type, homogenize_psf, n_coadd_psf):

    s1 = Sim(
        rng=np.random.RandomState(seed=10),
        gal_type=gal_type,
        psf_type=psf_type,
        homogenize_psf=homogenize_psf,
        n_coadd_psf=n_coadd_psf,
        n_coadd=10)

    s2 = Sim(
        rng=np.random.RandomState(seed=24357),
        gal_type=gal_type,
        psf_type=psf_type,
        homogenize_psf=homogenize_psf,
        n_coadd_psf=n_coadd_psf,
        n_coadd=10)

    mbobs1 = s1.get_mbobs()
    mbobs2 = s2.get_mbobs()

    assert not np.array_equal(mbobs1[0][0].image, mbobs2[0][0].image)
    assert not np.array_equal(mbobs1[0][0].noise, mbobs2[0][0].noise)
    if psf_type != 'gauss':
        assert not np.array_equal(
            mbobs1[0][0].psf.image, mbobs2[0][0].psf.image)


@pytest.mark.parametrize('gal_type', ['exp'])
@pytest.mark.parametrize('psf_type', ['gauss', 'ps'])
@pytest.mark.parametrize('homogenize_psf', [False, True])
@pytest.mark.parametrize('n_coadd_psf', [1, 2, 3])
def test_sim_seeding_shears(
        gal_type, psf_type, homogenize_psf, n_coadd_psf):

    s1 = Sim(
        rng=np.random.RandomState(seed=10),
        gal_type=gal_type,
        psf_type=psf_type,
        homogenize_psf=homogenize_psf,
        n_coadd_psf=n_coadd_psf,
        n_coadd=10,
        g1=0.02)

    s2 = Sim(
        rng=np.random.RandomState(seed=10),
        gal_type=gal_type,
        psf_type=psf_type,
        homogenize_psf=homogenize_psf,
        n_coadd_psf=n_coadd_psf,
        n_coadd=10,
        g1=-0.02)

    mbobs1 = s1.get_mbobs()
    mbobs2 = s2.get_mbobs()

    assert not np.array_equal(mbobs1[0][0].image, mbobs2[0][0].image)
    assert np.array_equal(mbobs1[0][0].noise, mbobs2[0][0].noise)
    assert np.array_equal(mbobs1[0][0].psf.image, mbobs2[0][0].psf.image)
