import numpy as np
import ngmix
import galsim
import logging

from .psf_homogenizer import PSFHomogenizer
from .ps_psf import PowerSpectrumPSF

LOGGER = logging.getLogger(__name__)


class Sim(dict):
    """A simple simulation for metadetect testing.

    Parameters
    ----------

    Methods
    -------
    """
    def __init__(
            self, *,
            rng, gal_type, psf_type,
            shear_scene=True,
            n_coadd=10,
            g1=0.02, g2=0.0,
            dim=225, buff=25,
            noise=4.0,
            nobj_per_10k=80000):
        self.rng = rng
        self.gal_type = gal_type
        self.psf_type = psf_type
        self.n_coadd = n_coadd
        self.g1 = g1
        self.g2 = g2
        self.shear_scene = shear_scene
        self.dim = dim
        self.buff = buff
        self.noise = noise / np.sqrt(self.n_coadd)
        self.nobj_per_10k = nobj_per_10k
        self.im_cen = (dim - 1) / 2

        # hard coded to the coadd DES value
        self.pixelscale = 0.263
        self.wcs = galsim.PixelScale(self.pixelscale)

        # frac of a single dimension that is used for drawing objects
        frac = 1.0 - self.buff * 2 / self.dim

        # half of the width of center of the patch that has objects
        self.pos_width = self.dim * frac * 0.5 * self.pixelscale

        # compute number of objects
        # we have a default of approximately 80000 objects per 10k x 10k coadd
        # this sim dims[0] * dims[1] but we only use frac * frac of the area
        # so the number of things we want is
        # dims[0] * dims[1] / 1e4^2 * 80000 * frac * frac
        self.nobj = int(
            self.dim * self.dim / 1e8 * self.nobj_per_10k *
            frac * frac)

        self.shear_mat = galsim.Shear(g1=self.g1, g2=self.g2).getMatrix()

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

        im = np.zeros((self.dim, self.dim), dtype='f8')

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
            assert x_ll >= 0 and x_ll < self.dim - _im.shape[1]
            assert y_ll >= 0 and y_ll < self.dim - _im.shape[0]
            dx = pos.x - (x_ll + (_im.shape[1] - 1)/2)
            dy = pos.y - (y_ll + (_im.shape[0] - 1)/2)
            dx *= self.pixelscale
            dy *= self.pixelscale
            stamp = obj.shift(dx=dx, dy=dy).drawImage(
                nx=_im.shape[1],
                ny=_im.shape[0],
                wcs=self.wcs,
                method=method)

            im[y_ll:y_ll+stamp.array.shape[0],
               x_ll:x_ll+stamp.array.shape[1]] += stamp.array

        im += self.rng.normal(scale=self.noise, size=im.shape)
        wt = im*0 + 1.0/self.noise**2
        bmask = np.zeros(im.shape, dtype='i4')
        noise = self.rng.normal(size=im.shape) / np.sqrt(wt)

        galsim_jac = self._get_local_jacobian(x=self.im_cen, y=self.im_cen)

        psf_obs = self.get_psf_obs(x=self.im_cen, y=self.im_cen)

        # im, noise, psf_img = self._homogenize_psf(im, noise)
        # psf_obs.set_image(psf_img)

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

    # def _homogenize_psf(self, im, noise):
    #     # homogenize the psf
    #     def _func(row, col):
    #         galsim_jac = self._get_local_jacobian(x=col, y=row)
    #         image = galsim.ImageD(ncol=17, nrow=17, wcs=galsim_jac)
    #         for psf in self._psfs:
    #             _image = galsim.ImageD(ncol=17, nrow=17, wcs=galsim_jac)
    #             _image = psf.draw(
    #                 x=int(col+0.5),
    #                 y=int(row+0.5),
    #                 image=_image)
    #             image += _image
    #         psf_im = image.array
    #         psf_im /= np.sum(psf_im)
    #         return psf_im
    #
    #     hmg = PSFHomogenizer(_func, im.shape, patch_size=25, sigma=0.25)
    #     him = hmg.homogenize_image(im)
    #     hnoise = hmg.homogenize_image(noise)
    #     psf_img = hmg.get_target_psf()
    #
    #     return him, hnoise, psf_img

    def _get_local_jacobian(self, *, x, y):
        return self.wcs.jacobian(
            image_pos=galsim.PositionD(x=x+1, y=y+1))

    def _get_dxdy(self):
        return self.rng.uniform(
            low=-self.pos_width,
            high=self.pos_width,
            size=2)

    def _get_gal_exp(self):
        flux = 10**(0.4 * (30 - 20))
        half_light_radius = 0.5

        obj = galsim.Sersic(
            half_light_radius=half_light_radius,
            n=1,
        ).withFlux(flux)

        return obj

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

        for i in range(self.nobj):
            # unsheared offset from center of image
            dx, dy = self._get_dxdy()

            # get the galaxy
            if self.gal_type == 'exp':
                gal = self._get_gal_exp()
            else:
                raise ValueError('gal_type "%s" not valid!' % self.gal_type)

            # compute the final image position
            if self.shear_scene:
                sdx, sdy = np.dot(self.shear_mat, np.array([dx, dy]))
            else:
                sdx = dx
                sdy = dy

            pos = galsim.PositionD(
                x=sdx / self.pixelscale + self.im_cen,
                y=sdy / self.pixelscale + self.im_cen)

            # get the PSF info
            _, _psf_wcs, _, _psf, _ = self._render_psf_image(
                x=pos.x, y=pos.y)

            # shear, shift, and then convolve the galaxy
            gal = gal.shear(g1=self.g1, g2=self.g2)
            gal = galsim.Convolve(gal, _psf)

            all_band_obj.append([gal])
            positions.append(pos)

        return all_band_obj, positions

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
            psf = galsim.Gaussian(fwhm=0.9)
            psf_im = psf.drawImage(nx=33, ny=33, wcs=_psf_wcs).array
            method = 'auto'
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
