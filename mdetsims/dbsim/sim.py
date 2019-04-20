import copy
import logging

import numpy as np

from .erins_code.descwl_sim import (
    PositionSampler,
    CatalogSampler,
    DESWLSim
)

LOGGER = logging.getLogger(__name__)


# based on esheldon/dbsim-config/dbsim-lsstgauss-sall-y5sx-02.yaml
SIM_CONF = {
    # 'survey_name': 'LSST',
    'density_fac': 1.0,
    # 'bands': ['g', 'r', 'i'],
    'image_size_arcmin': 1.3,
    'positions': {
        'type': 'uniform',
        'width': 1.0,
    },
    # 'psf': {
    #     'type': 'gauss',
    #     'fwhm': 0.8,
    # },
    # 'shear_all': True
}


class ErinsDBSim(object):
    """Erin's dbsim code wrapped here for my use.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG to use for drawing the objects.
    g1 : float, optional
        The simulated shear for the 1-axis.
    g2 : float, optional
        The simulated shear for the 2-axis.
    shear_scene : bool, optional
        Whether or not to shear the full scene.
    gal_kws : dict or None, optional
        Extra keyword arguments to use when building galaxy objects.
        'survey_name' : str
            The name of survey in all caps, e.g. 'DES', 'LSST'.
    psf_kws : dict or None, optional
        Extra keyword arguments to pass to the constructors for PSF objects.

    Methods
    -------
    get_mbobs()
        Make a simulated MultiBandObsList for metadetect.

    Attributes
    ----------
    area_sqr_arcmin : float
        The effective area simulated in square arcmin assuming the pixel
        scale is in arcsec.
    """
    def __init__(self, *, rng, shear_scene=True, g1=0.02, g2=0.0,
                 gal_kws=None, psf_kws=None):
        self.rng = rng
        self.gal_kws = gal_kws
        self.psf_kws = psf_kws
        self.g1 = g1
        self.g2 = g2
        self.shear_scene = shear_scene

        self.conf = copy.deepcopy(SIM_CONF)
        self.conf['shear'] = [g1, g2]
        self.conf['shear_all'] = shear_scene
        gal_kws = self.gal_kws or {}
        self.conf.update(gal_kws)
        self.conf.update({'psf': psf_kws})
        LOGGER.error("config: %s", self.conf)

        self.area_sqr_arcmin = self.conf['positions']['width']**2

        self.pos_sampler = PositionSampler(self.conf['positions'], self.rng)
        self.cat_sampler = CatalogSampler(self.conf, self.rng)
        self.sim = DESWLSim(
            self.conf,
            self.cat_sampler,
            self.pos_sampler,
            self.rng,
        )

    def get_mbobs(self):
        """Make a simulated MultiBandObsList for metadetect.

        Returns
        -------
        mbobs : MultiBandObsList
        """
        self.sim.make_obs()
        mbobs = self.sim.obs

        # we need to set the bmask for the fitting code
        for obslist in mbobs:
            for obs in obslist:
                obs.bmask = np.zeros(obs.image.shape, dtype='i4')

        return mbobs
