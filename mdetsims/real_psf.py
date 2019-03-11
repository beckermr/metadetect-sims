import logging
import tempfile
import os

from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import numpy as np
import galsim
import fitsio
import joblib

from .lhs import get_optimized_lhs_design

LOGGER = logging.getLogger(__name__)

GOOD_FFT_SIZES = np.sort(np.array([
    32, 64, 128, 256, 512, 768, 1024, 1280, 2048, 2304, 2560, 3072,
    3840, 4096, 5120, 6400, 6912, 7680, 8192], dtype=int))


class RealPSF(object):
    """A class to generate real PSFs using optics + atmosphere from a set of
    serialized PSF images.

    This class does PCA decomposition plus a Gaussian Process interpolation.

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
        self.n_photons = d['n_photons'][0]

        self.psf_images = d['images'][0]
        self.psf_locs = d['locs'][0]

        self._interp_psf()

    def _interp_psf(self):
        # pca
        nc = min(5, self.psf_locs.shape[0])
        self._pca = PCA(n_components=nc)
        yc = self._pca.fit_transform(
            self.psf_images.reshape(self.psf_images.shape[0], -1))

        # get scaled positions
        Xc = self.psf_locs / self.im_width

        # we need an estimate of the noise in the PCA comps for the fit.
        # howver, we know the number of photons used and have a mean
        # profile. thus we use this info to estimate the noise.
        # compute image variance from poisson stats
        im_var = np.abs(self._pca.mean_ * self.n_photons)
        # pca is linear and pixels are independent so variances add
        c_std = np.sqrt(np.dot(self._pca.components_**2, im_var))
        # normalize to unit sum over the image
        c_std /= self.n_photons

        # gaussian process it
        kernel = ConstantKernel() * RBF()
        self._gps = []
        import tqdm
        for ind in tqdm.trange(nc):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=c_std[ind]**2,
                n_restarts_optimizer=1)
            gp.fit(Xc, yc[:, ind])
            self._gps.append(gp)

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
        Xp = np.array([[pos.y, pos.x]]) / self.im_width
        c = np.array([[gp.predict(Xp)[0] for gp in self._gps]])
        rim = self._pca.inverse_transform(c)[0].reshape(
            self.psf_width, self.psf_width)
        return galsim.InterpolatedImage(galsim.ImageD(rim), scale=self.scale)


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
    n_lhs : int, optional
        The number of points to use the LHS design.

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
            n_lhs=512,
            n_photons=5e6):
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)
        self.base_deviate = galsim.BaseDeviate(self.rng.randint(1, 2**32-1))
        self.scale = scale
        self.n_lhs = n_lhs

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

        width = self.field_of_view * 60 * 60 * self.scale
        res = self.rng.uniform(
            low=-width,
            high=width,
            size=2) / self.scale
        self._cen_x = res[1]
        self._cen_y = res[0]

        self._build_atm()
        self._build_optics()

    def save_to_fits_serial(self, filename, rng=None, n_jobs=1):
        """Save a grid of PSF images to a file.

        Parameters
        ----------
        filename : str
            The file in which to save the PSF images.
        rng : np.random.RandomState or None, optional
            A numpy RNG to use. If None, then the rng attached to the class
            is used.
        n_jobs : int, optional
            The number of cores to use. The default of 1 results in purely
            serial execution.
        """
        rng = rng or self.rng
        psf_locs = get_optimized_lhs_design(
            2, self.n_lhs, rng=self.rng) * self.im_width
        seeds = rng.randint(1, 2**32-1, size=self.n_lhs).astype(int)
        # galsim chokes on numpy int64 types
        seeds = [int(s) for s in seeds]

        ims = np.zeros(
            (self.n_lhs, self.psf_width, self.psf_width),
            dtype='f4')

        import tqdm
        for i, s in tqdm.tqdm(enumerate(seeds), total=self.n_lhs):
            _rng = galsim.BaseDeviate(seed=s)
            psf_im = self.get_rec(
                row=psf_locs[i, 0],
                col=psf_locs[i, 1],
                rng=_rng)
            ims[i] = psf_im

        data = np.zeros(1, dtype=[
            ('im_width', 'i8'),
            ('psf_width', 'i8'),
            ('scale', 'f8'),
            ('n_photons', 'f8'),
            ('images', 'f8', ims.shape),
            ('locs', 'f8', psf_locs.shape)])
        data['im_width'] = self.im_width
        data['psf_width'] = self.psf_width
        data['n_photons'] = self.n_photons
        data['scale'] = self.scale
        data['images'][0] = ims
        data['locs'][0] = psf_locs

        fitsio.write(filename, data, clobber=True)

    def save_to_fits(self, filename, rng=None, n_jobs=1):
        """Save a grid of PSF images to a file.

        Parameters
        ----------
        filename : str
            The file in which to save the PSF images.
        rng : np.random.RandomState or None, optional
            A numpy RNG to use. If None, then the rng attached to the class
            is used.
        n_jobs : int, optional
            The number of cores to use. The default of 1 results in purely
            serial execution.
        """
        rng = rng or self.rng
        psf_locs = get_optimized_lhs_design(
            2, self.n_lhs, rng=self.rng) * self.im_width
        seeds = rng.randint(1, 2**32-1, size=self.n_lhs).astype(int)
        # galsim chokes on numpy int64 types
        seeds = [int(s) for s in seeds]

        def _measure_psf(_gen, seeds, xs, ys):
            if isinstance(_gen, str):
                _gen = joblib.load(_gen)
            ims = []
            import tqdm
            for seed, x, y in tqdm.tqdm(zip(seeds, xs, ys), total=len(seeds)):
                _rng = galsim.BaseDeviate(seed=seed)
                psf_im = self.get_rec(
                    row=y,
                    col=x,
                    rng=_rng)
                ims.append(psf_im)
            return ims, xs, ys

        # call once to create screens
        _measure_psf(self, [1], [0], [0])

        with tempfile.TemporaryDirectory() as tmpdir:
            if n_jobs > 1:
                # bundle to reduce overheads and predump the data
                n_per_job = int(np.ceil(
                    self.n_lhs / n_jobs))
                fname = os.path.join(tmpdir, 'data.pkl')
                joblib.dump(self, fname)
            else:
                # if we are using 1 core, then compute the phase screens once
                n_per_job = self.n_lhs
                fname = self

            jobs = []
            _xs = []
            _ys = []
            _seeds = []
            for loc in range(self.n_lhs):
                if len(_seeds) < n_per_job:
                    _xs.append(psf_locs[loc, 1])
                    _ys.append(psf_locs[loc, 0])
                    _seeds.append(seeds[loc])

                if len(_seeds) == n_per_job:
                    jobs.append(
                        joblib.delayed(_measure_psf)(
                            fname, _seeds, _xs, _ys))
                    _xs = []
                    _ys = []
                    _seeds = []

            if len(_seeds) > 0:
                jobs.append(
                    joblib.delayed(_measure_psf)(
                        fname, _seeds, _xs, _ys))

            outputs = joblib.Parallel(
                verbose=10,
                n_jobs=int(n_jobs),
                pre_dispatch='2*n_jobs')(jobs)

        # make sure they all get done
        assert sum(len(o[0]) for o in outputs) == self.n_lhs

        ims = np.zeros(
            (self.n_lhs, self.psf_width, self.psf_width),
            dtype='f4')
        loc = 0
        for psfs, _, _ in outputs:
            for psf in psfs:
                ims[loc] = psf
                loc += 1
        assert loc == self.n_lhs

        data = np.zeros(1, dtype=[
            ('im_width', 'i8'),
            ('psf_width', 'i8'),
            ('scale', 'f8'),
            ('n_photons', 'f8'),
            ('images', 'f8', ims.shape),
            ('locs', 'f8', psf_locs.shape)])
        data['im_width'] = self.im_width
        data['psf_width'] = self.psf_width
        data['n_photons'] = self.n_photons
        data['scale'] = self.scale
        data['images'][0] = ims
        data['locs'][0] = psf_locs

        fitsio.write(filename, data, clobber=True)

    def get_rec(self, *, row, col, rng=None, chunk_size=1e5):
        """Get an image of the PSF at a given (row, col).

        Parameters
        ----------
        row : float
            The row or y-value for the PSF location in zero-indexed image
            coordinates.
        col : float
            The column or x-value for the PSF location in zero-indexed image
            coordinates.
        rng : galsim.BaseDeviate or None, optional
            The galsim RNG to use for photon shooting.
        chunk_size : float, optional
            Split the photon shooting in chunks of this size to save
            memory.

        Returns
        -------
        psf : np.ndarray
            An image of the PSF.
        """
        psf = self.getPSF(galsim.PositionD(x=col, y=row))

        n_chunks = int(np.ceil(self.n_photons / chunk_size))
        im = np.zeros((self.psf_width, self.psf_width))
        n_done = 0

        for i in range(n_chunks):
            n_draw = chunk_size
            if n_done + n_draw > self.n_photons:
                n_draw = self.n_photons - n_done

            if n_draw <= 0:
                break

            _im = psf.drawImage(
                nx=self.psf_width,
                ny=self.psf_width,
                scale=self.scale,
                method='phot',
                n_photons=n_draw,
                rng=rng or self.base_deviate).array
            im += (_im * n_draw / self.n_photons)

            n_done += n_draw

        return im

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
        psf = self._screens.makePSF(
            lam=self.lam,
            exptime=self.exposure_time,
            diam=self.diam,
            obscuration=self.obscuration,
            geometric_shooting=True,
            theta=(
                (x * self.scale) * galsim.arcsec,
                (y * self.scale) * galsim.arcsec))

        return psf

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

        self._screens = galsim.Atmosphere(
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

        opt = galsim.OpticalScreen(
            lam_0=self.lam,
            diam=self.diam,
            obscuration=self.obscuration,
            **kwargs)

        # order them so I know where things are for later...
        _screens = galsim.PhaseScreenList()
        _screens.append(opt)
        for i in range(len(self._screens)):
            _screens.append(self._screens[i])
        self._screens = _screens


def get_good_fft_sizes(sizes):
    ind = np.searchsorted(GOOD_FFT_SIZES, sizes)
    ind = np.clip(ind, 0, len(GOOD_FFT_SIZES)-1)
    return GOOD_FFT_SIZES[ind]
