import numpy as np
import galsim


class PowerSpectrumPSF(object):
    """Produce a spatially varying Moffat PSF according to the power spectrum
    given by Heymans et al. (2012).

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance.
    im_width : float
        The width of the image in pixels.
    buff : int
        An extra buffer of pixels for things near the edge.
    scale : float
        The pixel scale of the image
    trunc : float
        The truncation scale for the shape/magnification power spectrum
        used to generate the PSF variation.
    noise_level : float, optional
        If not `None`, generate a noise field to add to the PSF images with
        desired noise. A value of 1e-2 generates a PSF image with an
        effective signal-to-noise of ~250.

    Methods
    -------
    getPSF(pos)
        Get a PSF model at a given position.
    """
    def __init__(self, *,
                 rng, im_width, buff, scale, trunc=1, noise_level=None):
        self._rng = rng
        self._im_cen = (im_width - 1)/2
        self._scale = scale
        self._tot_width = im_width + 2 * buff
        self._x_scale = 2.0 / self._tot_width / scale
        self._noise_level = noise_level
        self._buff = buff

        # set the power spectrum and PSF params
        # Heymans et al, 2012 found L0 ~= 3 arcmin, given as 180 arcsec here.
        def _pf(k):
            return (k**2 + (1./180)**2)**(-11./6.) * np.exp(-(k*trunc)**2)
        self._ps = galsim.PowerSpectrum(
            e_power_function=_pf,
            b_power_function=_pf)
        ng = 64
        gs = max(self._tot_width * self._scale / ng, 1)
        self.ng = ng
        self.gs = gs
        self._ps.buildGrid(
            grid_spacing=gs,
            ngrid=ng,
            get_convergence=True,
            variance=0.02**2,
            rng=galsim.BaseDeviate(self._rng.randint(1, 2**30)))

        if self._noise_level is not None and self._noise_level > 0:
            self._noise_field = self._rng.normal(
                size=(im_width + buff + 37, im_width + buff + 37)
            ) * noise_level

        def _getlogmnsigma(mean, sigma):
            logmean = np.log(mean) - 0.5*np.log(1 + sigma**2/mean**2)
            logvar = np.log(1 + sigma**2/mean**2)
            logsigma = np.sqrt(logvar)
            return logmean, logsigma

        lm, ls = _getlogmnsigma(0.9, 0.1)
        self._fwhm_central = np.exp(self._rng.normal() * ls + lm)

        ls = 0.005
        fac2 = 10
        fac3 = fac2 * 10
        self._fwhm_x = self._rng.normal() * ls
        self._fwhm_y = self._rng.normal() * ls
        self._fwhm_xx = self._rng.normal() * ls / fac2
        self._fwhm_xy = self._rng.normal() * ls / fac2
        self._fwhm_yy = self._rng.normal() * ls / fac2
        self._fwhm_xxx = self._rng.normal() * ls / fac3
        self._fwhm_xxy = self._rng.normal() * ls / fac3
        self._fwhm_xyy = self._rng.normal() * ls / fac3
        self._fwhm_yyy = self._rng.normal() * ls / fac3

    def _get_atm(self, x, y):
        xs = (x + 1 - self._im_cen) * self._scale
        ys = (y + 1 - self._im_cen) * self._scale
        g1, g2 = self._ps.getShear((xs, ys))
        mu = self._ps.getMagnification((xs, ys))

        xs = (x - self._im_cen) * self._x_scale
        ys = (y - self._im_cen) * self._x_scale
        fwhm = (
            self._fwhm_central +
            xs * self._fwhm_x +
            ys * self._fwhm_y +
            xs * xs * self._fwhm_xx +
            xs * ys * self._fwhm_xy +
            ys * ys * self._fwhm_yy +
            xs * xs * xs * self._fwhm_xxx +
            xs * xs * ys * self._fwhm_xxy +
            xs * ys * ys * self._fwhm_xyy +
            ys * ys * ys * self._fwhm_yyy
            )

        psf = galsim.Moffat(
            beta=2.5,
            fwhm=fwhm).lens(g1=g1, g2=g2, mu=mu)

        return psf

    def getPSF(self, pos):
        """Get a PSF model at a given position.

        Parameters
        ----------
        pos : galsim.PositionD
            The position at which to compute the PSF. In zero-indexed
            pixel coordinates.

        Returns
        -------
        psf : galsim.GSObject
            A representation of the PSF as a galism object.
        """
        psf = self._get_atm(pos.x, pos.y)

        if self._noise_level is not None and self._noise_level > 0:
            xll = int(pos.x + self._buff - 16)
            yll = int(pos.y + self._buff - 16)
            assert xll >= 0 and xll+33 <= self._noise_field.shape[1]
            assert yll >= 0 and yll+33 <= self._noise_field.shape[0]

            stamp = self._noise_field[yll:yll+33, xll:xll+33].copy()
            psf += galsim.InterpolatedImage(
                galsim.ImageD(stamp, scale=self._scale),
                normalization="sb")

        return psf
