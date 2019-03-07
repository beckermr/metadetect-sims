import logging
import numpy as np
import galsim
import fitsio

LOGGER = logging.getLogger(__name__)

GOOD_FFT_SIZES = np.sort(np.array([
    32, 64, 128, 256, 512, 768, 1024, 1280, 2048, 2304, 2560, 3072,
    3840, 4096, 5120, 6400, 6912, 7680, 8192], dtype=int))


class RealPSF(object):
    """A class to generate real PSFs using optics + atmosphere from a set of
    serialized PSF images.

    Parameters
    ----------
    filename : str
        The FITS file with the PSF images.

    Methods
    -------
    getPSF(pos)
        Get the PSF at a given position.
    """
    def __init__(self, filename):
        self.filename = filename
        d = fitsio.read(filename)
        self.im_width = d['im_width'][0]
        self.psf_width = d['psf_width'][0]
        self.scale = d['scale'][0]
        self.psf_images = d['flat_image'][0].reshape(
            self.im_width, self.im_width, self.psf_width, self.psf_width)

    def getPSF(self, pos):
        """Get a PSF model at a given position.

        Note the nearest integer position is used. One should always use
        `method = 'no_pixel'` when drawing with this image.

        Parameters
        ----------
        pos : galsim.PositionD
            The position at which to compute the PSF. In zero-indexed
            pixel coordinates.

        Returns
        -------
        psf : galsim.GSObject
            A representation of the PSF as a galism interpolated image.
        """
        x = int(np.clip(pos.x, 0, self.im_width-1))
        y = int(np.clip(pos.y, 0, self.im_width-1))
        psf = self.psf_images[y, x].copy()
        return galsim.InterpolatedImage(galsim.ImageD(psf), scale=self.scale)


class RealPSFGenerator(object):
    """A class to generate real PSFs using optics + atmosphere and serialize
    them to disk for later use.

    Parameters
    ----------
    seed : int
        A seed for the random number generator.
    scale : float
        The PSF image pixel scale.
    exposure_time : float, optional
        The length of the exposure in seconds.
    lam : float, optional
        The wavelength of the light in nanometers.
    diam : float, optional
        The diameter of the telescope in meters.
    obscuration : float, optional
        The linear fraction of the telescope aprture that is obscured.
    field_of_view : float, optional
        The size of the field of view in degrees.
    effective_r0_500 : float, optional
        The effective strength of the atmospheric turbulence. The smaller
        this number, the strong the turbulence and the larger the atmospheric
        component of the PSF. The default generates a DES-like PSF.
    im_width : int, optional
        The shape of the image on the focal plane. A PSF for each pixel in this
        image is saved to disk when `save_to_fits` is called.
    psf_width : int, optional
        The shape of the PSF image to save to disk. Generally you want this
        to be an odd number and relatively small.
    n_photons : int, optional
        The number of photos to use to draw the PSFs.

    Methods
    -------
    save_to_fits(filename)
        Save an array of PSF imags at each pixel in an image.
    getPSF(pos)
        Get the PSF at a given position.
    """
    def __init__(
            self, *,
            seed,
            scale,
            exposure_time=90.0,
            lam=700.0,
            diam=4.0,
            obscuration=0.42,
            field_of_view=2.2,
            effective_r0_500=0.1,
            im_width=225,
            psf_width=17,
            n_photons=5e6):
        self.rng = np.random.RandomState(seed=seed)
        self.base_deviate = galsim.BaseDeviate(self.rng.randint(1, 2**32-1))
        self.scale = scale

        self.exposure_time = exposure_time
        self.lam = lam
        self.diam = diam
        self.obscuration = obscuration
        self.field_of_view = field_of_view
        self.effective_r0_500 = effective_r0_500
        self.im_width = im_width
        self.psf_width = psf_width
        self.n_photons = n_photons
        self._im_cen = (im_width - 1) / 2

        width = self.field_of_view * 60 * 60 * 0.25
        res = self.rng.uniform(
            low=-width,
            high=width,
            size=2) / self.scale
        self._cen_x = res[1]
        self._cen_y = res[0]

        self._build_atm()
        self._build_optics()

    def save_to_fits(self, filename, rng=None):
        """Save a grid of PSF images to a file.

        Parameters
        ----------
        filename : str
            The file in which to save the PSF images.
        rng : np.random.RandomState or None, optional
            A numpy RNG to use. If None, then the BaseDeviate attached to the
            class is used.
        """
        ims = np.zeros(
            (self.im_width, self.im_width, self.psf_width, self.psf_width),
            dtype='f4')
        if rng is not None:
            seeds = rng.randint(1, 2**32-1, size=self.im_width**2).astype(int)

        for y in range(self.im_width):
            for x in range(self.im_width):
                if rng is not None:
                    # galsim chokes on numpy int64 types
                    _rng = galsim.BaseDeviate(
                        seed=int(seeds[y * self.im_width + x]))
                else:
                    _rng = None
                psf = self.getPSF(galsim.PositionD(x=x, y=y))
                psf_im = psf.drawImage(
                    nx=self.psf_width,
                    ny=self.psf_width,
                    scale=self.scale,
                    method='phot',
                    n_photons=self.n_photons,
                    rng=_rng or self.base_deviate)
                ims[y, x, :, :] = psf_im.array
        ims = ims.flatten()
        data = np.zeros(1, dtype=[
            ('im_width', 'i8'),
            ('psf_width', 'i8'),
            ('scale', 'f8'),
            ('flat_image', 'f4', ims.shape[0])])
        data['im_width'] = self.im_width
        data['psf_width'] = self.psf_width
        data['scale'] = self.scale
        data['flat_image'][0] = ims

        fitsio.write(filename, data, clobber=True)

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
        x = pos.x - self._im_cen + self._cen_x
        y = pos.y - self._im_cen + self._cen_y
        psf = self._atm.makePSF(
            lam=self.lam,
            exptime=self.exposure_time,
            diam=self.diam,
            obscuration=self.obscuration,
            theta=(
                (x * self.scale) * galsim.arcsec,
                (y * self.scale) * galsim.arcsec))

        return galsim.Convolve(self._optpsf, psf)

    def _build_atm(self):
        """Build an atmosphere following Jee and Tyson, galsim and my gut."""
        max_wind_speed = 20  # m/s

        max_screen_size = (
            self.exposure_time *
            max_wind_speed)  # m

        altitude = [0, 2.58, 5.16, 7.73, 12.89, 15.46]  # km
        max_screen_size = np.maximum(
            max_screen_size * 1.5,  # factor of 1.5 for periodic FFTs
            (max_screen_size + 2 * np.array(altitude) * 1e3 *
             self.field_of_view / 180 * np.pi)  # 2 fields of view
        )

        # draw a new effective r0 w/ 1% deviation around th fiducial value
        effr0 = self.rng.lognormal(np.log(self.effective_r0_500), 0.01)

        # draw weights with similar variations
        fid_weights = np.array([0.652, 0.172, 0.055, 0.025, 0.074, 0.022])
        weights = np.exp(
            self.rng.normal(size=len(fid_weights)) * 0.01 +
            np.log(fid_weights))
        weights /= np.sum(weights)

        # get effective screen r0's and the pixel scales
        screen_r0_500 = effr0 * np.power(weights, -3.0/5.0)
        screen_scales = np.clip(0.25 * screen_r0_500, 0.1, np.inf)

        # compute a good FFT size, clipping at max of 8192
        nominal_npix = np.clip(np.ceil(
            max_screen_size / screen_scales).astype(int), 0, 8192)
        npix = get_good_fft_sizes(nominal_npix)

        # for screens that need too many pixels,
        # we can either go back and make the overall size smaller
        # or make the pixels bigger
        # here I am going to make the size smaller since we want the proper
        # power but can tolerate some wrapping of the FFTs on average
        screen_sizes = screen_scales * npix

        LOGGER.debug('screen # of pixels: %s', npix)
        LOGGER.debug('screen pixel sizes: %s', screen_scales)
        LOGGER.debug('pixel size / r0_500: %s', screen_scales / screen_r0_500)
        LOGGER.debug('fraction of ideal screen size: %s',
                     screen_sizes / max_screen_size)

        speed = self.rng.uniform(0, max_wind_speed, size=6)  # m/s
        direction = [self.rng.uniform(0, 360) * galsim.degrees
                     for i in range(6)]

        self._atm = galsim.Atmosphere(
            r0_500=screen_r0_500,
            screen_size=screen_sizes,
            altitude=altitude,
            L0=25.0,
            speed=speed,
            direction=direction,
            screen_scale=screen_scales,
            rng=self.base_deviate)

    def _build_optics(self):
        # from galsim examples/great3/cgc.yaml
        rms_aberration = 0.26
        names = [
            "defocus", "astig1", "astig2", "coma1", "coma2",
            "trefoil1", "trefoil2", "spher"]
        weights = np.array(
            [0.13, 0.13, 0.14, 0.06, 0.06, 0.05, 0.06, 0.03])
        weights /= np.sqrt(np.sum(weights**2))
        weights *= rms_aberration
        kwargs = {
            k: a * self.rng.normal()
            for k, a in zip(names, weights)}

        self._optpsf = galsim.OpticalPSF(
            lam=self.lam,
            diam=self.diam,
            obscuration=self.obscuration,
            suppress_warning=False,
            **kwargs)


def get_good_fft_sizes(sizes):
    ind = np.searchsorted(GOOD_FFT_SIZES, sizes)
    ind = np.clip(ind, 0, len(GOOD_FFT_SIZES)-1)
    return GOOD_FFT_SIZES[ind]
