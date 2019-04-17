import logging
import os

import numpy as np

import ngmix
import galsim
import fitsio

from .psf_homogenizer import PSFHomogenizer
from .ps_psf import PowerSpectrumPSF
from .real_psf import RealPSF
from .masking import generate_bad_columns, generate_cosmic_rays
from .interp import interpolate_image_and_noise
from .symmetrize import symmetrize_bad_mask

LOGGER = logging.getLogger(__name__)


class Sim(dict):
    """A simple simulation for metadetect testing.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG to use for drawing the objects.
    gal_type : str
        The kind of galaxy to simulate.
    psf_type : str
        The kind of PSF to simulate.
    scale : float
        The pixel scale of the image.
    shear_scene : bool, optional
        Whether or not to shear the full scene.
    n_coadd : int, optional
        The number of single epoch images in a coadd. This number is used to
        scale the noise.
    n_coadd_psf : int, optional
        The number of PSF images to coadd for models with variable PSFs. The
        default of None uses the same number of PSFs as `n_coadd`.
    g1 : float, optional
        The simulated shear for the 1-axis.
    g2 : float, optional
        The simulated shear for the 2-axis.
    dim : int, optional
        The total dimension of the image.
    buff : int, optional
        The width of the buffer region.
    noise : float, optional
        The noise for a single epoch image.
    ngal : float, optional
        The number of objects to simulate per arcminute.
    gal_grid : int or None
        If not `None`, galaxies are laid out on a grid of `gal_grid` x
        `gal_grid` dimensions in the central part of the image.
    psf_kws : dict or None, optional
        Extra keyword arguments to pass to the constructors for PSF objects.
        See the doc strings of the PSF objects `PowerSpectrumPSF` and `RealPSF`
        for details.
    gal_kws : dict or None, optional
        Extra keyword arguments to use when building galaxy objects.

        For gal_type == 'wldeblend', these keywords can be

            'survey_name' : str
                The name of survey in all caps, e.g. 'DES', 'LSST'.
            'catalog' : str
                A path to the catalog to draw from. If this keyword is not
                given, you need to have the one square degree catsim catalog
                in the current working directory or in the directory given by
                the environment variable 'CATSIM_DIR'.

    homogenize_psf : bool, optional
        Apply PSF homogenization to the image.
    mask_and_interp : bool, optional
        Apply fake pixel masking for bad columns and cosmic rays and then
        interpolate them.

    Methods
    -------
    get_mbobs()
        Make a simulated MultiBandObsList for metadetect.
    get_psf_obs(*, x, y):
        Get an ngmix Observation of the PSF at the position (x, y).

    Attributes
    ----------
    area_sqr_arcmin : float
        The effective area simulated in square arcmin assuming the pixel
        scale is in arcsec.

    Notes
    -----
    The valid kinds of galaxies are

        'exp' : Sersic objects at very high s/n with n = 1
        'ground_galsim_parametric' : a typical ground-based sample
            from the galsim COSMOS catalog
        'wldeblend' : a sample drawn from the WeakLensingDeblending package

    The valid kinds of PSFs are

        'gauss' : a FWHM 0.9 arcsecond Gaussian
        'ps' : a PSF from power spectrum model for shape variation and
            cubic model for size variation
        'real' : a PSF model drawn randomly from a model of the atmosphere
            and optics in a set of files
        'piff' : a PSF model drawn randomly from a set of piff files
    """
    def __init__(
            self, *,
            rng, gal_type, psf_type,
            scale,
            shear_scene=True,
            n_coadd=1,
            n_coadd_psf=None,
            g1=0.02, g2=0.0,
            dim=225, buff=25,
            noise=180,
            ngal=45.0,
            gal_grid=None,
            psf_kws=None,
            gal_kws=None,
            homogenize_psf=False,
            mask_and_interp=False):
        self.rng = rng
        self.noise_rng = np.random.RandomState(seed=rng.randint(1, 2**32-1))
        self.gal_type = gal_type
        self.psf_type = psf_type
        self.n_coadd = n_coadd
        self.g1 = g1
        self.g2 = g2
        self.shear_scene = shear_scene
        self.dim = dim
        self.buff = buff
        self.noise = noise / np.sqrt(self.n_coadd)
        self.ngal = ngal
        self.gal_grid = gal_grid
        self.im_cen = (dim - 1) / 2
        self.psf_kws = psf_kws
        self.gal_kws = gal_kws
        self.n_coadd_psf = n_coadd_psf or n_coadd
        self.homogenize_psf = homogenize_psf
        self.mask_and_interp = mask_and_interp

        self.area_sqr_arcmin = ((self.dim - 2*self.buff) * scale / 60)**2

        self._galsim_rng = galsim.BaseDeviate(
            seed=self.rng.randint(low=1, high=2**32-1))

        # typical pixel scale
        self.scale = scale
        self.wcs = galsim.PixelScale(self.scale)

        # frac of a single dimension that is used for drawing objects
        frac = 1.0 - self.buff * 2 / self.dim

        # half of the width of center of the patch that has objects
        self.pos_width = self.dim * frac * 0.5 * self.scale

        # for wldeblend galaxies, we have to adjust some of the input
        # parameters since they are computed self consisently from the
        # input catalog and/or package defaults
        if self.gal_type == 'wldeblend':
            self._extra_init_for_wldeblend()

        # given the input number of objects to simulate per square arcminute,
        # compute the number we actually need
        self.nobj = int(
            self.ngal *
            (self.dim * self.scale / 60 * frac)**2)

        self.shear_mat = galsim.Shear(g1=self.g1, g2=self.g2).getMatrix()

        if self.gal_grid is not None:
            self.nobj = self.gal_grid * self.gal_grid

    def _extra_init_for_wldeblend(self):
        # gaurd the import here
        import descwl

        # make sure to find the proper catalog
        gal_kws = self.gal_kws or {}
        if 'catalog' not in gal_kws:
            fname = os.path.join(
                os.environ.get('CATSIM_DIR', '.'),
                'OneDegSq.fits')
        else:
            fname = gal_kws['catalog']

        self._wldeblend_cat = fitsio.read(fname)
        self._wldeblend_cat['pa_disk'] = self.rng.uniform(
            low=0.0, high=360.0, size=self._wldeblend_cat.size)
        self._wldeblend_cat['pa_bulge'] = self._wldeblend_cat['pa_disk']

        # set the survey name and exposure times
        if 'survey_name' not in gal_kws:
            survey_name = 'DES'
        else:
            survey_name = gal_kws['survey_name']

        if survey_name == 'DES':
            exptime = 90
        elif survey_name == 'LSST':
            exptime = 15
        else:
            exptime = None

        self._surveys = {}
        self._builders = {}
        noise_var = 0.0
        for band in ['g', 'r', 'i']:  # ['r', 'i', 'z']:
            # make the survey and code to build galaxies from it
            pars = descwl.survey.Survey.get_defaults(
                survey_name=survey_name,
                filter_band=band)

            pars['survey_name'] = survey_name
            pars['filter_band'] = band
            pars['pixel_scale'] = self.scale

            # note in the way we call the descwl package, the image width
            # and height is not actually used
            pars['image_width'] = self.dim
            pars['image_height'] = self.dim

            # the psf not actually used for anything given how we call the
            # descwl package
            pars['psf_model'] = galsim.Gaussian(fwhm=0.7)

            # we fix the exposure time and adjust the noise
            # pars['exposure_time'] = exptime

            self._surveys[band] = descwl.survey.Survey(**pars)
            self._builders[band] = descwl.model.GalaxyBuilder(
                survey=self._surveys[band],
                no_disk=False,
                no_bulge=False,
                no_agn=False,
                verbose_model=False)

            noise_var += self._surveys[band].mean_sky_level

        # now we reset the noise using the internal sky level appropriate
        # for the internal units
        # we are using stack of n_coadd total images equally
        # distributed over r i and z

        # rescale noise variance for a mean image in the bands
        # noise_var /= len(self._builders)

        # now we stack n_coadd / n_bands of those
        # self.noise = np.sqrt(noise_var / (self.n_coadd / len(self._builders)))
        self.noise = np.sqrt(noise_var / len(self._builders))

        # when we sample from the catalog, we need to pull the right number
        # of objects. Since the default catalog is one square degree
        # and we fill a fraction of the image, we need to set the
        # base source density `ngal`. This is in units of number per
        # square arcminute.
        self.ngal = self._wldeblend_cat.size / (60 * 60)

    def get_mbobs(self):
        """Make a simulated MultiBandObsList for metadetect.

        Returns
        -------
        mbobs : MultiBandObsList
        """
        all_band_obj, positions = self._get_band_objects()

        mbobs = ngmix.MultiBandObsList()

        _, _, _, _, method = self._render_psf_image(
            x=self.im_cen, y=self.im_cen)

        im = galsim.ImageD(nrow=self.dim, ncol=self.dim, xmin=0, ymin=0)

        band_objects = [o[0] for o in all_band_obj]
        for obj, pos in zip(band_objects, positions):
            # draw with setup_only to get the image size
            _im = obj.drawImage(
                wcs=self.wcs,
                method=method,
                setup_only=True).array
            assert _im.shape[0] == _im.shape[1]

            # now get location of the stamp
            x_ll = int(pos.x - (_im.shape[1] - 1)/2)
            y_ll = int(pos.y - (_im.shape[0] - 1)/2)

            # get the offset of the center
            dx = pos.x - (x_ll + (_im.shape[1] - 1)/2)
            dy = pos.y - (y_ll + (_im.shape[0] - 1)/2)
            dx *= self.scale
            dy *= self.scale

            # draw and set the proper origin
            stamp = obj.shift(dx=dx, dy=dy).drawImage(
                nx=_im.shape[1],
                ny=_im.shape[0],
                wcs=self.wcs,
                method=method)
            stamp.setOrigin(x_ll, y_ll)

            # intersect and add to total image
            overlap = stamp.bounds & im.bounds
            im[overlap] += stamp[overlap]

        im = im.array.copy()

        im += self.noise_rng.normal(scale=self.noise, size=im.shape)
        wt = im*0 + 1.0/self.noise**2
        bmask = np.zeros(im.shape, dtype='i4')
        noise = self.noise_rng.normal(size=im.shape) / np.sqrt(wt)

        if self.mask_and_interp:
            im, noise, bmask = self._mask_and_interp(im, noise)

        galsim_jac = self._get_local_jacobian(x=self.im_cen, y=self.im_cen)

        psf_obs = self.get_psf_obs(x=self.im_cen, y=self.im_cen)

        if self.homogenize_psf:
            im, noise, psf_img = self._homogenize_psf(im, noise)
            psf_obs.set_image(psf_img)

        jac = ngmix.jacobian.Jacobian(
            row=self.im_cen,
            col=self.im_cen,
            wcs=galsim_jac)

        obs = ngmix.Observation(
            im,
            weight=wt,
            bmask=bmask,
            jacobian=jac,
            psf=psf_obs,
            noise=noise)

        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

        return mbobs

    def _mask_and_interp(self, image, noise):
        LOGGER.debug('applying masking and interpolation')

        # here we make the mask
        bad_mask = generate_bad_columns(image.shape, rng=self.noise_rng)
        bad_mask |= generate_cosmic_rays(image.shape, rng=self.noise_rng)

        # applies a 90 degree rotation to make it symmetric
        symmetrize_bad_mask(bad_mask)

        # now we inteprolate the pixels in the noise and image field
        # that are masked
        _im, _nse = interpolate_image_and_noise(
            image=image,
            noise=noise,
            bad_mask=bad_mask,
            rng=self.noise_rng)

        return _im, _nse, bad_mask.astype(np.int32)

    def _homogenize_psf(self, im, noise):
        LOGGER.info('applying PSF homogenization')

        def _func(row, col):
            psf_im, _, _, _, _ = self._render_psf_image(
                x=col,
                y=row)
            return psf_im

        hmg = PSFHomogenizer(_func, im.shape, patch_size=25, sigma=0.25)
        him = hmg.homogenize_image(im)
        hnoise = hmg.homogenize_image(noise)
        psf_img = hmg.get_target_psf()

        return him, hnoise, psf_img

    def _get_local_jacobian(self, *, x, y):
        return self.wcs.jacobian(
            image_pos=galsim.PositionD(x=x+1, y=y+1))

    def _get_dxdy(self):
        if self.gal_grid is not None:
            yind, xind = np.unravel_index(
                self._gal_grid_ind, (self.gal_grid, self.gal_grid))
            dg = self.pos_width * 2 / self.gal_grid
            self._gal_grid_ind += 1
            return (
                yind * dg + dg/2 - self.pos_width,
                xind * dg + dg/2 - self.pos_width)
        else:
            return self.rng.uniform(
                low=-self.pos_width,
                high=self.pos_width,
                size=2)

    def _get_nobj(self):
        if self.gal_grid is not None:
            return self.nobj
        else:
            return self.rng.poisson(self.nobj)

    def _get_gal_exp(self):
        flux = 10**(0.4 * (30 - 18))
        half_light_radius = 0.5

        obj = galsim.Sersic(
            half_light_radius=half_light_radius,
            n=1,
        ).withFlux(flux)

        return obj

    def _get_gal_ground_galsim_parametric(self):
        if not hasattr(self, '_cosmo_cat'):
            self._cosmo_cat = galsim.COSMOSCatalog(sample='25.2')
        angle = self.rng.uniform() * 360
        gal = self._cosmo_cat.makeGalaxy(
            gal_type='parametric',
            rng=self._galsim_rng
        ).rotate(
            angle * galsim.degrees
        ).withScaledFlux(
            (4.0**2 * (1.0 - 0.42**2)) /
            (2.4**2 * (1.0 - 0.33**2)) *
            90
        )
        return gal

    def _get_gal_wldeblend(self):
        rind = self.rng.choice(self._wldeblend_cat.size)
        angle = self.rng.uniform() * 360

        # we divide by the number of bands here since we are averaging over
        # n_coadd / n_bands images per band and then normalizing by n_coadd
        gal = galsim.Sum(
            [self._builders[band].from_catalog(
                self._wldeblend_cat[rind], 0, 0,
                self._surveys[band].filter_band).model
             for band in self._builders]) / len(self._builders)

        # apply an extra rotation
        gal = gal.rotate(angle * galsim.degrees)

        return gal

    def _get_band_objects(self):
        """Get a list of effective PSF-convolved galsim images w/ their
        offsets in the image.

        Returns
        -------
        all_band_objs : list of lists
            A list of lists of objects in each band.
        positions : list of galsim.PositionD
            A list of galsim positions for each object.
        """
        all_band_obj = []
        positions = []

        nobj = self._get_nobj()

        if self.gal_grid is not None:
            self._gal_grid_ind = 0

        for i in range(nobj):
            # unsheared offset from center of image
            dx, dy = self._get_dxdy()

            # get the galaxy
            if self.gal_type == 'exp':
                gal = self._get_gal_exp()
            elif self.gal_type == 'ground_galsim_parametric':
                gal = self._get_gal_ground_galsim_parametric()
            elif self.gal_type == 'wldeblend':
                gal = self._get_gal_wldeblend()
            else:
                raise ValueError('gal_type "%s" not valid!' % self.gal_type)

            # compute the final image position
            if self.shear_scene:
                sdx, sdy = np.dot(self.shear_mat, np.array([dx, dy]))
            else:
                sdx = dx
                sdy = dy

            pos = galsim.PositionD(
                x=sdx / self.scale + self.im_cen,
                y=sdy / self.scale + self.im_cen)

            # get the PSF info
            _, _psf_wcs, _, _psf, _ = self._render_psf_image(
                x=pos.x, y=pos.y)

            # shear, shift, and then convolve the galaxy
            gal = gal.shear(g1=self.g1, g2=self.g2)
            gal = galsim.Convolve(gal, _psf)

            all_band_obj.append([gal])
            positions.append(pos)

        return all_band_obj, positions

    def _stack_ps_psfs(self, *, x, y, **kwargs):
        if not hasattr(self, '_psfs'):
            self._psfs = [
                PowerSpectrumPSF(
                    rng=self.rng,
                    im_width=self.dim,
                    buff=self.dim/2,
                    scale=self.scale,
                    **kwargs)
                for _ in range(self.n_coadd_psf)]
            LOGGER.debug('stacking %d power spectrum psfs', self.n_coadd_psf)

        _psf_wcs = self._get_local_jacobian(x=x, y=y)

        psf = galsim.Sum([
            p.getPSF(galsim.PositionD(x=x, y=y))
            for p in self._psfs]).withFlux(1)
        psf_im = psf.drawImage(nx=21, ny=21, wcs=_psf_wcs).array.copy()
        psf_im /= np.sum(psf_im)

        return psf, psf_im

    def _stack_real_psfs(self, *, x, y, filenames):
        if not hasattr(self, '_psfs'):
            fnames = self.rng.choice(
                filenames, size=self.n_coadd_psf, replace=False)
            self._psfs = [RealPSF(fname) for fname in fnames]
            LOGGER.debug('stacking %d real psfs', self.n_coadd_psf)

        _psf_wcs = self._get_local_jacobian(x=x, y=y)

        psf = galsim.Sum([
            p.getPSF(galsim.PositionD(x=x, y=y))
            for p in self._psfs]).withFlux(1)
        psf_im = psf.drawImage(
            nx=21, ny=21, wcs=_psf_wcs, method='no_pixel').array.copy()
        psf_im /= np.sum(psf_im)

        return psf, psf_im

    def _stack_piff_psfs(self, *, x, y, filenames):
        # import so we don't require piff to run the code
        import piff

        if not hasattr(self, '_psfs'):
            fnames = self.rng.choice(
                filenames, size=self.n_coadd_psf, replace=False)
            self._psfs = [piff.PSF.read(fname) for fname in fnames]
            LOGGER.debug('stacking %d piff psfs', self.n_coadd_psf)

        wcs = self._get_local_jacobian(x=x, y=y)

        image = galsim.ImageD(ncol=17, nrow=17, wcs=wcs)
        for psf in self._psfs:
            _image = galsim.ImageD(ncol=17, nrow=17, wcs=wcs)
            _image = psf.draw(
                x=int(x+0.5),
                y=int(y+0.5),
                image=_image)
            image += _image
        psf_im = image.array
        psf_im /= np.sum(psf_im)

        psf = galsim.InterpolatedImage(galsim.ImageD(psf_im), wcs=wcs)

        return psf, psf_im

    def _render_psf_image(self, *, x, y):
        """Render the PSF image.

        Returns
        -------
        psf_image : array-like
            The pixel-convolved (i.e. effective) PSF.
        psf_wcs : galsim.JacobianWCS
            The WCS as a local Jacobian at the PSF center.
        noise : float
            An estimate of the noise in the image.
        psf_gs : galsim.GSObject
            The PSF as a galsim object.
        method : str
            Method to use to render images using this PSF.
        """
        _psf_wcs = self._get_local_jacobian(x=x, y=y)

        if self.psf_type == 'gauss':
            kws = self.psf_kws or {}
            fwhm = kws.get('fwhm', 0.9)
            psf = galsim.Gaussian(fwhm=fwhm)
            psf_im = psf.drawImage(nx=21, ny=21, wcs=_psf_wcs).array.copy()
            psf_im /= np.sum(psf_im)
            method = 'auto'
        elif self.psf_type == 'ps':
            kws = self.psf_kws or {}
            psf, psf_im = self._stack_ps_psfs(x=x, y=y, **kws)
            method = 'auto'
        elif self.psf_type == 'real':
            kws = self.psf_kws or {}
            psf, psf_im = self._stack_real_psfs(x=x, y=y, **kws)
            method = 'no_pixel'
        elif self.psf_type == 'piff':
            kws = self.psf_kws or {}
            psf, psf_im = self._stack_piff_psfs(x=x, y=y, **kws)
            method = 'no_pixel'
        else:
            raise ValueError('psf_type "%s" not valid!' % self.psf_type)

        # set the signal to noise to about 500
        target_s2n = 500.0
        target_noise = np.sqrt(np.sum(psf_im ** 2) / target_s2n**2)

        return psf_im, _psf_wcs, target_noise, psf, method

    def get_psf_obs(self, *, x, y):
        """Get an ngmix Observation of the PSF at a position.

        Parameters
        ----------
        x : float
            The column of the PSF.
        y : float
            The row of the PSF.

        Returns
        -------
        psf_obs : ngmix.Observation
            An Observation of the PSF.
        """
        psf_image, psf_wcs, noise, _, _ = self._render_psf_image(x=x, y=y)

        weight = np.zeros_like(psf_image) + 1.0/noise**2

        cen = (np.array(psf_image.shape) - 1.0)/2.0
        j = ngmix.jacobian.Jacobian(
            row=cen[0], col=cen[1], wcs=psf_wcs)
        psf_obs = ngmix.Observation(
            psf_image,
            weight=weight,
            jacobian=j)

        return psf_obs
