import logging
import os
import functools

import numpy as np

import ngmix
import galsim
import fitsio
import scipy.spatial

from .psf_homogenizer import PSFHomogenizer
from .ps_psf import PowerSpectrumPSF
from .masking import generate_bad_columns, generate_cosmic_rays
from .interp import interpolate_image_and_noise
from .cs_interp import interpolate_image_and_noise_cs
from .symmetrize import symmetrize_bad_mask
from .defaults import WLDEBLEND_DES_FACTOR, WLDEBLEND_LSST_FACTOR

LOGGER = logging.getLogger(__name__)


@functools.lru_cache(maxsize=8)
def _cached_catalog_read(fname):
    return fitsio.read(fname)


class Sim(object):
    """A simple simulation for metadetect testing.

    Parameters
    ----------
    OneSq_cat: fits file or None
        The whole redshift info for OneSq_cat
    frac_ex : float, 0 to 1
        The fraction of galxies with extra shear.
    rng : np.random.RandomState
        An RNG to use for drawing the objects.
    sel_rng : np.random.RandomState
        An independent RNG to use for determining the shear of objects.
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
    n_coadd_msk : int, optional
        A number to scale the masking effects. The default is 1.
    g1 : float, optional
        The simulated shear for the 1-axis.
    g2 : float, optional
        The simulated shear for the 2-axis.
    g1ex : float, optional
        The simulated shear for the 1-axis to another group of galaxies.
    g2ex : float, optional
        The simulated shear for the 2-axis to another group of galaxies.
    dim : int, optional
        The total dimension of the image.
    buff : int, optional
        The width of the buffer region.
    noise : float, optional
        The noise for a single epoch image.
    ngal : float, optional
        The number of objects to simulate per arcminute.
    n_bands : int, optional
        The number of bands to simulate.
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

        For any gal_type, on can add

            'min_dist' : float
                Minimum distance in arcseconds between the generatd object
                centers.

    homogenize_psf : bool, optional
        Apply PSF homogenization to the image.
    mask_and_interp : bool, optional
        Apply fake pixel masking for bad columns and cosmic rays and then
        interpolate them.
    add_bad_columns : bool, optional
        If False, do not add bad columns. Otherwise they will be added when
        `mask_and_interp` is True.
    add_cosmic_rays : bool, optional
        If False, do not add cosmic rays. Otherwise they will be added when
        `mask_and_interp` is True.
    bad_columns_kws : dict, optional
        A set of keyword arguments to pass to the bad column generator.
    interpolation_type : str, optional
        One of

            'cubic' : a 2d cubic interpolation
            'cs-fourier' : a Fourier basis compressed sensing interpolant

        The default is 'cubic'.
    ngal_factor : float, optional
        A factor to change the number density in the sims. It is set to 0.6
        automatically when using the wldeblend galaxy type for DES and 0.45
        when using this type for LSST.

    Methods
    -------
    get_mbobs()
        Make a simulated MultiBandObsList for metadetect.
    get_psf_obs(*, x, y, n_bands, band):
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
    """
    def __init__(
            self, *, OneSq_cat=None,
            frac_ex=0, rng, sel_rng,
            gal_type, psf_type, scale,
            shear_scene=True,
            n_coadd=1,
            n_coadd_psf=None,
            n_coadd_msk=1,
            g1=0.02, g2=0.0,
            g1ex=0.02, g2ex=0.0,
            dim=225, buff=25,
            noise=180,
            ngal=45.0,
            n_bands=1,
            gal_grid=None,
            psf_kws=None,
            gal_kws=None,
            homogenize_psf=False,
            mask_and_interp=False,
            add_bad_columns=True,
            add_cosmic_rays=True,
            bad_columns_kws=None,
            interpolation_type='cubic',
            ngal_factor=None):
        self.rng = rng
        self.OneSq_cat = OneSq_cat
        self.sel_rng = sel_rng
        self.frac_ex = frac_ex
        if self.frac_ex == 0:
            self.flag_ex = False
        else:
            self.flag_ex = True
        self.noise_rng = np.random.RandomState(seed=rng.randint(1, 2**32-1))
        self.gal_type = gal_type
        self.psf_type = psf_type
        self.n_coadd = n_coadd
        self.n_coadd_msk = n_coadd_msk
        self.g1 = g1
        self.g2 = g2
        self.g1ex = g1ex
        self.g2ex = g2ex
        self.shear_scene = shear_scene
        self.dim = dim
        self.buff = buff
        self.ngal_factor = ngal_factor
        # assumed to be one band unless otherwise specified via the
        # galaxy type
        self.noise = [noise / np.sqrt(self.n_coadd)] * n_bands
        self.ngal = ngal
        self.gal_grid = gal_grid
        self.im_cen = (dim - 1) / 2
        self.psf_kws = psf_kws
        self.gal_kws = gal_kws
        self.n_coadd_psf = n_coadd_psf or self.n_coadd
        self.homogenize_psf = homogenize_psf
        self.mask_and_interp = mask_and_interp
        self.add_bad_columns = add_bad_columns
        self.add_cosmic_rays = add_cosmic_rays
        self.bad_columns_kws = bad_columns_kws or {}
        self.interpolation_type = interpolation_type

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
        if self.ngal_factor is None:
            self.ngal_factor = 1
        LOGGER.info('ngal adjustment factor: %f', self.ngal_factor)

        self.nobj = int(
            self.ngal * self.ngal_factor *
            (self.dim * self.scale / 60 * frac)**2)

        self.shear_mat = galsim.Shear(g1=self.g1, g2=self.g2).getMatrix()
        self.shear_matex = galsim.Shear(g1=self.g1ex, g2=self.g2ex).getMatrix()

        if self.gal_grid is not None:
            self.nobj = self.gal_grid * self.gal_grid

        self.n_bands = len(self.noise)

        LOGGER.info('simulating %d bands', self.n_bands)

    def _extra_init_for_wldeblend(self):
        # guard the import here
        import descwl

        # make sure to find the proper catalog
        gal_kws = self.gal_kws or {}
        if 'catalog' not in gal_kws:
            fname = os.path.join(
                os.environ.get('CATSIM_DIR', '.'),
                'OneDegSq.fits')
        else:
            fname = gal_kws['catalog']

        self._wldeblend_cat = _cached_catalog_read(fname)
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
            if self.n_coadd != 10:
                LOGGER.warning(
                    'simulating DES with descwl - '
                    'input n_coadd != 10!')
        elif survey_name == 'LSST':
            LOGGER.debug(
                'simulating LSST with descwl - ignoring n_coadd input!')
            exptime = None
        else:
            raise ValueError("Survey '%s' is not valid!" % survey_name)

        bands = gal_kws.get('bands', ['r', 'i', 'z'])
        LOGGER.debug('simulating bands: %s', bands)

        self._surveys = []
        self._builders = []
        noises = []
        for iband, band in enumerate(bands):
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

            # reset the exposure times if we want
            if survey_name == 'DES':
                pars['exposure_time'] = exptime * self.n_coadd

            # some versions take in the PSF and will complain if it is not
            # given
            try:
                _svy = descwl.survey.Survey(**pars)
            except Exception:
                pars['psf_model'] = None
                _svy = descwl.survey.Survey(**pars)

            self._surveys.append(_svy)
            self._builders.append(descwl.model.GalaxyBuilder(
                survey=self._surveys[iband],
                no_disk=False,
                no_bulge=False,
                no_agn=False,
                verbose_model=False))

            noises.append(np.sqrt(self._surveys[iband].mean_sky_level))

        self.noise = noises

        # when we sample from the catalog, we need to pull the right number
        # of objects. Since the default catalog is one square degree
        # and we fill a fraction of the image, we need to set the
        # base source density `ngal`. This is in units of number per
        # square arcminute.
        self.ngal = self._wldeblend_cat.size / (60 * 60)

        # we use a factor of 0.6 to make sure the depth matches that in
        # the real data
        if self.ngal_factor is None:
            if survey_name == 'DES':
                self.ngal_factor = WLDEBLEND_DES_FACTOR
            elif survey_name == 'LSST':
                self.ngal_factor = WLDEBLEND_LSST_FACTOR
            else:
                raise ValueError("Survey '%s' is not valid!" % survey_name)

        LOGGER.info('catalog density: %f per sqr arcmin', self.ngal)

    def get_mbobs(self, return_truth_cat=False):
        """Make a simulated MultiBandObsList for metadetect.

        Parameters
        ----------
        return_truth_cat : bool
            If True, return the truth catalog.

        Returns
        -------
        mbobs : MultiBandObsList
        """

        all_band_obj, positions, z_population = self._get_band_objects()
        truth_cat = np.zeros(len(positions), dtype=[('x', 'f8'), ('y', 'f8'),
                                                    ('z_population', 'f8')])
        truth_cat['z_population'] = z_population

        _, _, _, _, method = self._render_psf_image(
            x=self.im_cen, y=self.im_cen)

        mbobs = ngmix.MultiBandObsList()

        for band in range(self.n_bands):

            im = galsim.ImageD(nrow=self.dim, ncol=self.dim, xmin=0, ymin=0)

            band_objects = [o[band] for o in all_band_obj]
            for obj_ind, (obj, pos) in enumerate(zip(band_objects, positions)):
                truth_cat['x'][obj_ind] = pos.x
                truth_cat['y'][obj_ind] = pos.y

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

            im += self.noise_rng.normal(scale=self.noise[band], size=im.shape)
            wt = im*0 + 1.0/self.noise[band]**2
            bmask = np.zeros(im.shape, dtype='i4')
            noise = self.noise_rng.normal(size=im.shape) / np.sqrt(wt)

            if self.mask_and_interp:
                im, noise, bmask = self._mask_and_interp(im, noise)

            galsim_jac = self._get_local_jacobian(x=self.im_cen, y=self.im_cen)

            psf_obs = self.get_psf_obs(
                x=self.im_cen, y=self.im_cen, band=band)

            if self.homogenize_psf:
                im, noise, psf_img = self._homogenize_psf(
                    im, noise, band)
                psf_obs.set_image(psf_img)

            jac = ngmix.jacobian.Jacobian(
                row=self.im_cen,
                col=self.im_cen,
                wcs=galsim_jac)

            obs = ngmix.Observation(
                im,
                weight=wt,
                bmask=bmask,
                ormask=bmask.copy(),
                jacobian=jac,
                psf=psf_obs,
                noise=noise)

            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

        if return_truth_cat:
            return mbobs, truth_cat
        else:
            return mbobs

    def _mask_and_interp(self, image, noise):
        LOGGER.debug('applying masking and interpolation')

        # here we make the mask
        bad_mask = np.zeros(image.shape, dtype=np.bool)
        if self.add_bad_columns:
            bad_mask |= generate_bad_columns(
                image.shape, rng=self.noise_rng,
                mean_bad_cols=self.n_coadd_msk,
                **self.bad_columns_kws)
        if self.add_cosmic_rays:
            bad_mask |= generate_cosmic_rays(
                image.shape, rng=self.noise_rng,
                mean_cosmic_rays=self.n_coadd_msk)

        # applies a 90 degree rotation to make it symmetric
        symmetrize_bad_mask(bad_mask)

        # muck the image
        image[bad_mask] = 1e12

        # now we inteprolate the pixels in the noise and image field
        # that are masked
        if self.interpolation_type == 'cs-fourier':
            _im, _nse = interpolate_image_and_noise_cs(
                image=image,
                noise=noise,
                bad_mask=bad_mask,
                rng=self.noise_rng,
                c=50,
                sampling_rate=1)
        elif self.interpolation_type == 'cubic':
            _im, _nse = interpolate_image_and_noise(
                image=image,
                noise=noise,
                bad_mask=bad_mask,
                rng=self.noise_rng)
        else:
            raise ValueError(
                'interpolation "%s" is not defined' % self.interpolation_type)

        return _im, _nse, bad_mask.astype(np.int32)

    def _homogenize_psf(self, im, noise, band):
        LOGGER.info('applying PSF homogenization')

        def _func(row, col):
            psf_im, _, _, _, _ = self._render_psf_image(
                x=col,
                y=row)
            return psf_im[band]

        hmg = PSFHomogenizer(_func, im.shape, patch_size=25, sigma=0.25)
        him = hmg.homogenize_image(im)
        hnoise = hmg.homogenize_image(noise)
        psf_img = hmg.get_target_psf()

        return him, hnoise, psf_img

    def _get_local_jacobian(self, *, x, y):
        return self.wcs.jacobian(
            image_pos=galsim.PositionD(x=x+1, y=y+1))

    def _get_dxdy(self, others=None, min_dist=0):
        if self.gal_grid is not None:
            yind, xind = np.unravel_index(
                self._gal_grid_ind, (self.gal_grid, self.gal_grid))
            dg = self.pos_width * 2 / self.gal_grid
            self._gal_grid_ind += 1
            return (
                yind * dg + dg/2 - self.pos_width,
                xind * dg + dg/2 - self.pos_width)
        else:
            while True:
                dx, dy = self.rng.uniform(
                    low=-self.pos_width,
                    high=self.pos_width,
                    size=2)

                if others is not None:
                    tree = scipy.spatial.cKDTree(others)
                    d, _ = tree.query(np.array([dx, dy]))
                    if np.min(d) > min_dist:
                        LOGGER.debug('min nbr dist: %f', np.min(d))
                        break
                else:
                    break
            return dx, dy

    def _get_nobj(self):
        if self.gal_grid is not None:
            return self.nobj
        else:
            return self.rng.poisson(self.nobj)

    def _get_gal_exp(self):
        flux = 10**(0.4 * (30 - 18))
        half_light_radius = 0.5

        _gal = []
        for _ in range(self.n_bands):
            obj = galsim.Sersic(
                half_light_radius=half_light_radius,
                n=1,
            ).withFlux(flux)
            _gal.append(obj)

        return _gal

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
        return [gal for _ in range(self.n_bands)]

    def _get_gal_wldeblend(self):
        rind = self.rng.choice(self._wldeblend_cat.size)
        angle = self.rng.uniform() * 360

        gals = [
            self._builders[band].from_catalog(
                self._wldeblend_cat[rind], 0, 0,
                self._surveys[band].filter_band).model.rotate(
                    angle * galsim.degrees)
            for band in range(len(self._builders))]

        return gals, rind

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

        # turn on/off varying shear
        if self.gal_type == 'wldeblend':
            z_population = np.zeros(nobj)
        else:
            z_population = self.sel_rng.choice(2, nobj, p=[self.frac_ex,
                                                           1-self.frac_ex])

        if self.gal_grid is not None:
            self._gal_grid_ind = 0

        gal_kws = self.gal_kws or {}
        if 'min_dist' in gal_kws:
            LOGGER.debug('using min dist: %f', gal_kws['min_dist'])
            others = []

        for i in range(nobj):
            # unsheared offset from center of image
            if 'min_dist' in gal_kws:
                if i == 0:
                    dx, dy = self._get_dxdy()
                else:
                    dx, dy = self._get_dxdy(
                        others=np.array(others),
                        min_dist=gal_kws['min_dist'])
                others.append([dx, dy])
            else:
                dx, dy = self._get_dxdy()

            # get the galaxy
            if self.gal_type == 'exp':
                gals = self._get_gal_exp()
            elif self.gal_type == 'ground_galsim_parametric':
                gals = self._get_gal_ground_galsim_parametric()
            elif self.gal_type == 'wldeblend':
                gals, rind = self._get_gal_wldeblend()

                # add shear info to calculate the final position
                if 0.2 < self.OneSq_cat[rind] < 0.3:
                    z_population[i] = 1
                else:
                    z_population[i] = 0
            else:
                raise ValueError('gal_type "%s" not valid!' % self.gal_type)

            # compute the final image position
            if self.shear_scene:
                if self.flag_ex:
                    if z_population[i] == 1:
                        sdx, sdy = np.dot(self.shear_mat, np.array([dx, dy]))
                    else:
                        sdx, sdy = np.dot(self.shear_matex, np.array([dx, dy]))
                else:
                    sdx, sdy = np.dot(self.shear_mat, np.array([dx, dy]))
            else:
                sdx = dx
                sdy = dy

            pos = galsim.PositionD(
                x=sdx / self.scale + self.im_cen,
                y=sdy / self.scale + self.im_cen)

            # get the PSF info
            _, _psf_wcs, _, _psfs, _ = self._render_psf_image(
                x=pos.x, y=pos.y)

            # shear, shift, and then convolve the galaxy
            _obj = []
            for gal, _psf in zip(gals, _psfs):
                if self.flag_ex:
                    if z_population[i] == 1:
                        gal = gal.shear(g1=self.g1, g2=self.g2)
                    else:
                        gal = gal.shear(g1=self.g1ex, g2=self.g2ex)
                else:
                    gal = gal.shear(g1=self.g1, g2=self.g2)
                gal = galsim.Convolve(gal, _psf)
                _obj.append(gal)

            all_band_obj.append(_obj)
            positions.append(pos)

        return all_band_obj, positions, z_population

    def _get_psf_box_size(self, psfs, _psf_wcs):
        if not hasattr(self, '_cached_psf_box_size'):
            max_box_size = -1
            for psf in psfs:
                psf_im = psf.drawImage(wcs=_psf_wcs, setup_only=True).array
                _box_size = np.max(psf_im.shape)
                if _box_size % 2 == 0:
                    _box_size += 1
                if _box_size > max_box_size:
                    max_box_size = _box_size
            self._cached_psf_box_size = max_box_size
            LOGGER.debug('psf box size set to %d', self._cached_psf_box_size)

        return self._cached_psf_box_size

    def _stack_ps_psfs(self, *, x, y, **kwargs):
        if not hasattr(self, '_psfs'):
            self._psfs = [[
                    PowerSpectrumPSF(
                        rng=self.rng,
                        im_width=self.dim,
                        buff=self.dim/2,
                        scale=self.scale,
                        **kwargs)
                    for _ in range(self.n_coadd_psf)]
                for _ in range(self.n_bands)]

            LOGGER.debug('stacking %d power spectrum psfs', self.n_coadd_psf)

        _psf_wcs = self._get_local_jacobian(x=x, y=y)

        test_psfs = []
        for i in range(self.n_bands):
            test_psfs += [
                p.getPSF(galsim.PositionD(x=x, y=y))
                for p in self._psfs[i]]

        psf_dim = self._get_psf_box_size(test_psfs, _psf_wcs)

        psfs = []
        psf_ims = []
        for i in range(self.n_bands):
            psf = galsim.Sum([
                p.getPSF(galsim.PositionD(x=x, y=y))
                for p in self._psfs[i]]).withFlux(1)
            psf_im = psf.drawImage(
                nx=psf_dim, ny=psf_dim, wcs=_psf_wcs).array.copy()
            psf_im /= np.sum(psf_im)

            psfs.append(psf)
            psf_ims.append(psf_im)

        return psfs, psf_ims

    def _get_wldeblend_psfs(self, *, x, y):
        _psf_wcs = self._get_local_jacobian(x=x, y=y)
        psf_dim = self._get_psf_box_size(
            [self._surveys[i].psf_model for i in range(len(self._surveys))],
            _psf_wcs)

        psfs = []
        psf_ims = []
        for i in range(len(self._surveys)):
            psfs.append(self._surveys[i].psf_model)
            psf_im = psfs[-1].drawImage(
                nx=psf_dim, ny=psf_dim, wcs=_psf_wcs).array.copy()
            psf_im /= np.sum(psf_im)
            psf_ims.append(psf_im)

        return psfs, psf_ims

    def _render_psf_image(self, *, x, y):
        """Render the PSF image.

        Returns
        -------
        psf_images : list of array-like
            The pixel-convolved (i.e. effective) PSF images.
        psf_wcs : galsim.JacobianWCS
            The WCS as a local Jacobian at the PSF center.
        noises : list of floats
            Estimates of the noise in the images.
        psf_gs : list of galsim.GSObject
            The PSFs as galsim objects.
        method : str
            Method to use to render images using this PSF.
        """
        _psf_wcs = self._get_local_jacobian(x=x, y=y)

        if self.psf_type == 'gauss':
            kws = self.psf_kws or {}
            fwhm = kws.get('fwhm', 0.9)
            LOGGER.debug('gaussian PSF FWHM is %f', fwhm)
            psf = galsim.Gaussian(fwhm=fwhm)
            psf_dim = self._get_psf_box_size([psf], _psf_wcs)
            psf_im = psf.drawImage(
                nx=psf_dim, ny=psf_dim, wcs=_psf_wcs).array.copy()
            psf_im /= np.sum(psf_im)
            method = 'auto'
            psf_ims = [psf_im] * self.n_bands
            psfs = [psf] * self.n_bands
        elif self.psf_type == 'ps':
            kws = self.psf_kws or {}
            psfs, psf_ims = self._stack_ps_psfs(
                x=x, y=y, **kws)
            method = 'auto'
        elif self.psf_type == 'wldeblend':
            psfs, psf_ims = self._get_wldeblend_psfs(x=x, y=y)
            method = 'auto'
        else:
            raise ValueError('psf_type "%s" not valid!' % self.psf_type)

        # set the signal to noise to about 500
        target_s2n = 500.0
        target_noises = np.sqrt(
            [np.sum(psf_im ** 2) for psf_im in psf_ims]) / target_s2n

        return psf_ims, _psf_wcs, target_noises, psfs, method

    def get_psf_obs(self, *, x, y, band):
        """Get an ngmix Observation of the PSF at a position.

        Parameters
        ----------
        x : float
            The column of the PSF.
        y : float
            The row of the PSF.
        band : int
            The index of the desired band.

        Returns
        -------
        psf_obs : ngmix.Observation
            An Observation of the PSF.
        """
        psf_images, psf_wcs, noises, _, _ = self._render_psf_image(
            x=x, y=y)

        weight = np.zeros_like(psf_images[band]) + 1.0/noises[band]**2

        cen = (np.array(psf_images[band].shape) - 1.0)/2.0
        j = ngmix.jacobian.Jacobian(
            row=cen[0], col=cen[1], wcs=psf_wcs)
        psf_obs = ngmix.Observation(
            psf_images[band],
            weight=weight,
            jacobian=j)

        return psf_obs
