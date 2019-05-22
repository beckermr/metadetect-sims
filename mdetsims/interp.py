import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator

import numba
from numba import njit

# fast code from numba due to @sheldon


@njit
def _get_nearby_good_pixels(image, bad_msk, nbad, buff=4):
    """
    get the set of good pixels surrounding bad pixels.

    Parameters
    ----------
    image: array
        The image data
    bad_msk: bool array
        2d array of mask bits.  True means it is a bad
        pixel

    Returns
    -------
    bad_pix:
        bad pix is the set of bad pixels, shape [nbad, 2]
    good_pix:
        good pix is the set of bood pixels around the bad
        pixels, shape [ngood, 2]
    good_im:
        the set of good image values, shape [ngood]
    good_ind:
        the 1d indices of the good pixels row*ncol + col
    """

    nrows, ncols = bad_msk.shape

    ngood = nbad*(2*buff+1)**2
    good_pix = np.zeros((ngood, 2), dtype=numba.int64)
    good_ind = np.zeros(ngood, dtype=numba.int64)
    bad_pix = np.zeros((ngood, 2), dtype=numba.int64)
    good_im = np.zeros(ngood, dtype=image.dtype)

    ibad = 0
    igood = 0
    for row in range(nrows):
        for col in range(ncols):
            val = bad_msk[row, col]
            if val:
                bad_pix[ibad] = (row, col)
                ibad += 1

                row_start = row - buff
                row_end = row + buff
                col_start = col - buff
                col_end = col + buff

                if row_start < 0:
                    row_start = 0
                if row_end > (nrows-1):
                    row_end = nrows-1
                if col_start < 0:
                    col_start = 0
                if col_end > (ncols-1):
                    col_end = ncols-1

                for rc in range(row_start, row_end+1):
                    for cc in range(col_start, col_end+1):
                        tval = bad_msk[rc, cc]
                        if not tval:

                            if igood == ngood:
                                raise RuntimeError('good_pix too small')

                            # got a good one, add it to the list
                            good_pix[igood] = (rc, cc)
                            good_im[igood] = image[rc, cc]

                            # keep track of index
                            ind = rc*ncols + cc
                            good_ind[igood] = ind
                            igood += 1

    bad_pix = bad_pix[:ibad, :]

    good_pix = good_pix[:igood, :]
    good_ind = good_ind[:igood]
    good_im = good_im[:igood]

    return bad_pix, good_pix, good_im, good_ind


def _grid_interp(*, image, bad_msk):
    """
    interpolate the bad pixels in an image

    Parameters
    ----------
    image: array
        the pixel data
    bad_msk: array
        boolean array, True means it is a bad pixel
    """

    nrows, ncols = image.shape
    npix = bad_msk.size

    nbad = bad_msk.sum()
    bm_frac = nbad/npix

    if bm_frac < 0.90:

        bad_pix, good_pix, good_im, good_ind = \
            _get_nearby_good_pixels(image, bad_msk, nbad)

        # extract unique ones
        gi, ind = np.unique(good_ind, return_index=True)

        good_pix = good_pix[ind, :]
        good_im = good_im[ind]

        img_interp = CloughTocher2DInterpolator(
            good_pix,
            good_im,
            fill_value=0.0,
        )
        interp_image = image.copy()
        interp_image[bad_msk] = img_interp(bad_pix)

        return interp_image

    else:
        return None


def interpolate_image_and_noise(
        *, image, noise, bad_mask, rng):
    """Interpolate an image and a noise field using the
    `scipy.interpolate.CloughTocher2DInterpolator`.

    Parameters
    ----------
    image : array-like
        The image to interpolate.
    noise : array-like
        The noise field to interpolate.
    bad_mask : array-like
        A boolean mask indicating which pixels to interpolate.
    rng : `numpy.random.RandomState`
        An RNG instance to use.

    Returns
    -------
    interp_image : array-like
        The interpolated image.
    interp_noise : array-like
        The interpolated noise image.
    """
    if np.any(bad_mask):
        interp_image = _grid_interp(image=image, bad_msk=bad_mask)
        if interp_image is None:
            return None, None

        interp_noise = _grid_interp(image=noise, bad_msk=bad_mask)
        if interp_noise is None:
            return None, None

        return interp_image, interp_noise
    else:
        # return a copy here since the caller expects new images
        return image.copy(), noise.copy()
