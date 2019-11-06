import numpy as np

from .lanczos import lanczos_resample_three, lanczos_resample_one


def invert_affine_transform_wcs(u, v, wcs):
    """Invert a galsim.AffineTransform WCS.

    The AffineTransform WCS forward model is

        [u, v] = Jac * ([x, y] - origin) + world_origin

    where the `*` is a matrix multiplication and `[u, v]` is a column
    vector, etc.

    Parameters
    ----------
    u : np.ndarray
        The first world coordinate value.
    v : np.ndarray
        The second world coordinate value.
    wcs : galsim.AffineTransform
        The AffineTransform WCS object to invert.

    Returns
    -------
    x : np.ndarray
        The first image coordinate value.
    y : np.ndarray
        The second image coordinate value.
    """
    invmat = np.linalg.inv(
        np.array([[wcs.dudx, wcs.dudy], [wcs.dvdx, wcs.dvdy]]))
    du = u - wcs.u0
    dv = v - wcs.v0
    x = invmat[0, 0] * du + invmat[0, 1] * dv + wcs.x0
    y = invmat[1, 0] * du + invmat[1, 1] * dv + wcs.y0
    return x, y


def coadd_image_noise_interpfrac(
        se_images, se_noises, se_interp_fracs, se_wcs_objs,
        coadd_wgts, coadd_scale, coadd_dim):
    """Coadd a set of SE images, noise fields, and interpolation fractions.

    Parameters
    ----------
    se_images : list of np.ndarray
        The list of SE images to coadd.
    se_noises : list of np.ndarray
        The list of SE noise images to coadd.
    se_interp_fracs : list of np.ndarray
        The list of SE interpolated fraction images to coadd.
    se_wcs_objs : list of galsim.BaseWCS or children
        The WCS objects for each of the SE images.
    coadd_wgts : 1d array-like object of floats
        The relative coaddng weights for each of the SE images.
    coadd_scale : float
        The pixel scale of desired coadded image.
    coadd_dim : int
        The number of pixels desired for the final coadd image..

    Returns
    -------
    img : np.ndarray, shape (coadd_dim, coadd_dim)
        The coadd image.
    nse : np.ndarray, shape (coadd_dim, coadd_dim)
        The coadd noise image.
    intp : np.ndarray, shape (coadd_dim, coadd_dim)
        The interpolated flux fraction in each coadd pixel.
    """

    # coadd pixel coords
    y, x = np.mgrid[0:coadd_dim, 0:coadd_dim]
    u = x.ravel() * coadd_scale
    v = y.ravel() * coadd_scale

    coadd_image = np.zeros((coadd_dim, coadd_dim), dtype=np.float64)
    coadd_noise = np.zeros((coadd_dim, coadd_dim), dtype=np.float64)
    coadd_intp = np.zeros((coadd_dim, coadd_dim), dtype=np.float32)

    wgts = coadd_wgts / np.sum(coadd_wgts)

    for se_im, se_nse, se_intp, se_wcs, wgt in zip(
            se_images, se_noises, se_interp_fracs, se_wcs_objs, wgts):

        se_x, se_y = invert_affine_transform_wcs(u, v, se_wcs)
        im, nse, intp, _ = lanczos_resample_three(
            se_im / se_wcs.pixelArea(),
            se_nse / se_wcs.pixelArea(),
            se_intp,
            se_y,
            se_x)

        coadd_image += (im.reshape((coadd_dim, coadd_dim)) * wgt)
        coadd_noise += (nse.reshape((coadd_dim, coadd_dim)) * wgt)
        coadd_intp += (intp.reshape((coadd_dim, coadd_dim)) * wgt)

    coadd_image *= (coadd_scale**2)
    coadd_noise *= (coadd_scale**2)

    return coadd_image, coadd_noise, coadd_intp


def coadd_psfs(
        se_psfs, se_wcs_objs, coadd_wgts,
        coadd_scale, coadd_dim, coadd_offset, se_offsets):
    """Coadd the PSFs.

    Parameters
    ----------
    se_psfs : list of np.ndarray
        The list of SE PSF images to coadd.
    se_wcs_objs : list of galsim.BaseWCS or children
        The WCS objects for each of the SE PSFs.
    coadd_wgts : 1d array-like object of floats
        The relative coaddng weights for each of the SE PSFs.
    coadd_scale : float
        The pixel scale of desired coadded PSF image.
    coadd_dim : int
        The number of pixels desired for the final coadd PSF.
    coadd_offset : float
        The offset in pixels of the start of the coadd PSF image stamp.
    se_offsets : list of tuples of floats
        The offset in the SE image coords of the start of the SE PSF
        image.

    Returns
    -------
    psf : np.ndarray
        The coadded PSF image.
    """

    # coadd pixel coords
    y, x = np.mgrid[0:coadd_dim, 0:coadd_dim]
    u = (coadd_offset + x.ravel()) * coadd_scale
    v = (coadd_offset + y.ravel()) * coadd_scale

    coadd_image = np.zeros((coadd_dim, coadd_dim), dtype=np.float64)

    wgts = coadd_wgts / np.sum(coadd_wgts)

    for se_psf, se_wcs, wgt, se_offset in zip(
            se_psfs, se_wcs_objs, wgts, se_offsets):

        se_x, se_y = invert_affine_transform_wcs(u, v, se_wcs)
        se_x -= se_offset[0]
        se_y -= se_offset[1]
        im, _ = lanczos_resample_one(se_psf / se_wcs.pixelArea(), se_y, se_x)
        coadd_image += (im.reshape((coadd_dim, coadd_dim)) * wgt)
    coadd_image *= (coadd_scale**2)

    return coadd_image
