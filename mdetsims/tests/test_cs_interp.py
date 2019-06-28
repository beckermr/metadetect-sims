import numpy as np
from ..cs_interp import interpolate_image_and_noise_cs


def test_interpolate_image_and_noise_cs():
    # linear image interp should be pretty good
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
    iimage, inoise = interpolate_image_and_noise_cs(
            image=image,
            noise=noise,
            bad_mask=bad_mask,
            rng=rng,
            c=1,
            sampling_rate=1.0)

    meets_tol = np.allclose(iimage, 10 + x*5, rtol=0, atol=0.01)
    if not meets_tol:
        print(np.max(np.abs(iimage - 10 - x*5)))
    assert meets_tol
    assert np.all(np.isfinite(iimage))
    assert np.all(np.isfinite(inoise))

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=422)
    noise = rng.normal(size=image.shape)
    assert not np.allclose(noise[bad_mask], inoise[bad_mask])
    assert np.allclose(noise[~bad_mask], inoise[~bad_mask])
