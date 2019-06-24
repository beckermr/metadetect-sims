import logging
import numpy as np
import esutil as eu

from .base_fitter import FitterBase, _fit_one_psf
from .util import Namer
from .bootstrappers import MomentsMetacalBootstrapper, MaxMetacalBootstrapper

logger = logging.getLogger(__name__)

METACAL_TYPES = ['noshear', '1p', '1m', '2p', '2m']


class MetacalFitter(FitterBase):
    """Run metacal on all a list of observations.

    Parameters
    ----------
    conf : dict
        A configuration dictionary.
    nband : int
        The number of bands.
    rng : np.random.RandomState
        An RNG instance.
    mof_fitter : mdetsims.metcal.MOFFitter
        An instantiated MOFFitter for doing neighbor subtraction.

    Methods
    -------
    go(mbobs_list_input)
    """
    def __init__(self, conf, nband, rng, mof_fitter=None, **kw):

        self.mof_fitter = mof_fitter

        super().__init__(conf, nband, rng)

    def _setup(self):
        self.metacal_prior = self._get_prior(self['metacal'])
        if self['metacal']['model'] != 'wmom':
            assert self.metacal_prior is not None
        self['metacal']['symmetrize_weight'] = self['metacal'].get(
            'symmetrize_weight', False)
        self['metacal']['keep_all_fits'] = self['metacal'].get(
            'keep_all_fits', False)

        # be safe friends!
        if 'types' in self['metacal']:
            assert self['metacal']['types'] == METACAL_TYPES

    @property
    def result(self):
        """Get the result data"""
        if not hasattr(self, '_result'):
            raise RuntimeError('run go() first')

        if self._result is not None:
            res = {}
            for key in self._result:
                if self._result[key] is not None:
                    res[key] = self._result[key].copy()
                else:
                    res[key] = None
            return res
        else:
            return None

    def go(self, mbobs_list):
        """Run metcal on a list of MultiBandObsLists.

        Parameters
        ----------
        mbobs_list: list of MultiBandObsList
            One for each object.  If it is a simple MultiBandObsList it will
            be converted to a list
        """
        if not isinstance(mbobs_list, list):
            mbobs_list = [mbobs_list]

        if self.mof_fitter is not None:
            # for mof fitting, we expect a list of mbobs_lists
            mof_data, epochs_data = self.mof_fitter.go(mbobs_list)
            if mof_data is None or epochs_data is None:
                self._result = None
                return None
            fitter = self.mof_fitter.get_mof_fitter()

            # test if the fitter worked
            res = fitter.get_result()
            if res['flags'] != 0:
                self._result = None
                return None

            # this gets all objects, all bands in a list of MultiBandObsList
            mbobs_list_mcal = fitter.make_corrected_obs()
        else:
            mbobs_list_mcal = mbobs_list
            mof_data = None
        self.mbobs_list_mcal = mbobs_list_mcal

        if self['metacal']['symmetrize_weight']:
            self._symmetrize_weights(mbobs_list_mcal)

        self._result = self._do_all_metacal(mbobs_list_mcal, data=mof_data)

    def _symmetrize_weights(self, mbobs_list):
        def _symmetrize_weight(wt):
            assert wt.shape[0] == wt.shape[1]
            wt_rot = np.rot90(wt)
            wzero = wt_rot == 0.0
            if np.any(wzero):
                wt[wzero] = 0.0

        for mbobs in mbobs_list:
            for obslist in mbobs:
                for obs in obslist:
                    _symmetrize_weight(obs.weight)

    def _do_all_metacal(self, mbobs_list, data=None):
        """run metacal on all objects

        NOTE: failed mbobs will have no entries in the final list
        """

        assert len(mbobs_list[0]) == self.nband

        result = {key: [] for key in METACAL_TYPES}
        for i, mbobs in enumerate(mbobs_list):
            if self._check_flags(mbobs):
                boot = self._do_one_metacal(mbobs)
                if isinstance(boot, dict):
                    res = boot
                else:
                    res = boot.get_metacal_result()

                # record the things
                if (res['mcal_flags'] != 0 and
                        not self['metacal']['keep_all_fits']):
                    logger.debug('a metacal fit failed - ignoring everything!')
                else:
                    arr_res = self._get_metacal_output(res, self.nband, mbobs)

                    if data is not None:
                        odata = data[i:i+1]
                        for key in METACAL_TYPES:
                            fit_data = eu.numpy_util.add_fields(
                                arr_res[key],
                                odata.dtype.descr)
                            eu.numpy_util.copy_fields(odata, fit_data)
                            arr_res[key] = fit_data

                    for key in result:
                        result[key].append(arr_res[key])

        for key in result:
            if len(result[key]) > 0:
                result[key] = eu.numpy_util.combine_arrlist(result[key])
            else:
                result[key] = None

        return result

    def _do_one_metacal(self, mbobs):
        conf = self['metacal']
        if conf['model'] == 'wmom':
            boot = self._get_bootstrapper(mbobs)

            boot.fit_metacal(
                rng=self.rng,
                moments_pars=conf,
                metacal_pars=conf['metacal_pars']
            )
        else:
            psf_pars = conf['psf']
            max_conf = conf['max_pars']

            tpsf_obs = mbobs[0][0].psf
            if not tpsf_obs.has_gmix():
                _fit_one_psf(tpsf_obs, psf_pars)

            psf_Tguess = tpsf_obs.gmix.get_T()

            boot = self._get_bootstrapper(mbobs)
            if 'lm_pars' in psf_pars:
                psf_fit_pars = psf_pars['lm_pars']
            else:
                psf_fit_pars = None

            prior = self.metacal_prior
            guesser = None

            boot.fit_metacal(
                psf_pars['model'],
                conf['model'],
                max_conf['pars'],
                psf_Tguess,
                psf_fit_pars=psf_fit_pars,
                psf_ntry=psf_pars['ntry'],
                prior=prior,
                guesser=guesser,
                ntry=max_conf['ntry'],
                metacal_pars=conf['metacal_pars'],
            )

        return boot

    def _check_flags(self, mbobs):
        """
        only one epoch, so anything that hits an edge
        """
        flags = self['metacal'].get('bmask_flags', None)

        if flags is not None:
            for obslist in mbobs:
                for obs in obslist:
                    msk = (obs.bmask & flags) != 0
                    if np.any(msk):
                        logger.info("   EDGE HIT")
                        return False

        return True

    def _print_result(self, data):
        logger.debug(
            "    mcal s2n: %g Trat: %g",
            data['mcal_s2n_noshear'][0],
            data['mcal_T_ratio_noshear'][0])

    def _get_metacal_dtype(self, npars, nband, mtype):
        dt = [
            ('x', 'f8'),
            ('y', 'f8'),
        ]

        n = Namer(front='mcal')
        if mtype == 'noshear':
            dt += [
                (n('psf_g'), 'f8', 2),
                (n('psf_T'), 'f8'),
            ]
        dt += [
            (n('flags'), 'i4'),
            (n('nfev'), 'i4'),
            (n('s2n'), 'f8'),
            (n('pars'), 'f8', npars),
            (n('pars_cov'), 'f8', (npars, npars)),
            (n('g'), 'f8', 2),
            (n('g_cov'), 'f8', (2, 2)),
            (n('T'), 'f8'),
            (n('T_err'), 'f8'),
            (n('T_ratio'), 'f8'),
            (n('flux'), 'f8', nband),
            (n('flux_cov'), 'f8', (nband, nband)),
            (n('flux_err'), 'f8', nband),
        ]

        return dt

    def _get_metacal_output(self, allres, nband, mbobs):
        # assume one epoch and line up in all
        # bands
        assert len(mbobs[0]) == 1, 'one epoch only'

        # needed for the code below
        assert METACAL_TYPES[0] == 'noshear'

        npars = len(allres['noshear']['pars'])
        n = Namer(front='mcal')
        arr_res = {}

        for mtype in METACAL_TYPES:
            res = allres[mtype]

            dt = self._get_metacal_dtype(npars, nband, mtype)
            data = np.zeros(1, dtype=dt)

            data0 = data[0]
            data0['y'] = mbobs[0][0].meta['orig_row']
            data0['x'] = mbobs[0][0].meta['orig_col']

            if mtype == 'noshear':
                if 'gpsf' in res:
                    data0[n('psf_g')] = res['gpsf']
                    data0[n('psf_T')] = res['Tpsf']
                else:
                    data0[n('psf_g')] = res['psf_g']
                    data0[n('psf_T')] = res['psf_T']

            for name in res:
                nn = n(name)
                if nn in data.dtype.names:
                    data0[nn] = res[name]

            # do this so it has the data for the if statment below
            arr_res[mtype] = data

            if 'T_ratio' in res:
                # correct the ratio of the moments to match mcal cuts
                data0[n('T_ratio')] = res['T_ratio'] / (1.2/0.5)
                logger.debug("scaling T ratio by %f", 1.0 / (1.2/0.5))
            else:
                # this relies on noshear coming first in the metacal
                # types
                data0[n('T_ratio')] = (
                    data0[n('T')]/arr_res['noshear']['mcal_psf_T'])

            # do it again because blah
            arr_res[mtype] = data

        return arr_res

    def _get_bootstrapper(self, mbobs):
        if self['metacal']['model'] == 'wmom':
            return MomentsMetacalBootstrapper(
                mbobs,
                verbose=False,
            )
        else:
            return MaxMetacalBootstrapper(
                mbobs,
                verbose=False,
            )
