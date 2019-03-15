import numpy as np
from ..interp import interpolate_image_and_noise


def test_interpolate_image_and_noise():
    # linear image interp should be perfect for regions smaller than the
    # patches used for interpolation
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    bad_mask = np.zeros_like(image, dtype=np.int32)
    bad_mask[30:35, 40:45] = 1
    bad_mask = bad_mask.astype(bool)

    rng = np.random.RandomState(seed=422)
    noise = rng.normal(size=image.shape)

    # put nans here to make sure interp is done ok
    image[bad_mask] = np.nan
    noise[bad_mask] = np.nan

    rng = np.random.RandomState(seed=42)
    iimage, inoise = interpolate_image_and_noise(
            image=image,
            noise=noise,
            bad_mask=bad_mask,
            rng=rng)

    assert np.allclose(iimage, 10 + x*5)
    assert np.all(np.isfinite(iimage))
    assert np.all(np.isfinite(inoise))

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=422)
    noise = rng.normal(size=image.shape)
    assert not np.allclose(noise[bad_mask], inoise[bad_mask])
    assert np.allclose(noise[~bad_mask], inoise[~bad_mask])


def test_interpolate_image_and_noise_big_missing():
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    bad_mask = np.zeros_like(image, dtype=np.int32)
    bad_mask[15:80, 15:80] = 1
    bad_mask = bad_mask.astype(bool)

    rng = np.random.RandomState(seed=422)
    noise = rng.normal(size=image.shape)

    # put nans here to make sure interp is done ok
    image[bad_mask] = np.nan
    noise[bad_mask] = np.nan

    rng = np.random.RandomState(seed=42)
    iimage, inoise = interpolate_image_and_noise(
        image=image,
        noise=noise,
        bad_mask=bad_mask,
        rng=rng)

    # interp will be waaay off but shpuld have happened
    assert np.all(np.isfinite(iimage))
    assert np.all(np.isfinite(inoise))

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=422)
    noise = rng.normal(size=image.shape)
    assert not np.allclose(noise[bad_mask], inoise[bad_mask])
    assert np.allclose(noise[~bad_mask], inoise[~bad_mask])
