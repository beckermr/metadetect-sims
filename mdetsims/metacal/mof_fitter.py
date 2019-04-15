import logging
import numpy as np

import esutil as eu
import ngmix
from ngmix.gexceptions import BootPSFFailure


from .util import Namer, NoDataError
from . import procflags
from .base_fitter import (
    FitterBase, get_stamp_guesses, _fit_all_psfs, _measure_all_psf_fluxes)

import mof

logger = logging.getLogger(__name__)


class MOFFitter(FitterBase):
    """A multi-object fitter.

    Parameters
    ----------
    conf : dict
        A configuration dictionary.
    nband : int
        The number of bands.
    rng : np.random.RandomState
        An RNG instance.

    Methods
    -------
    go(mbobs_list, ntry=2)
        Run the multi-object fitter.
    """
    def __init__(self, *args, **kw):

        super().__init__(*args, **kw)

        self.mof_prior = self._get_prior(self['mof'])
        self._set_mof_fitter_class()
        self._set_guess_func()

    def go(self, mbobs_list, ntry=2):
        """Run the multi object fitter

        Parameters
        ----------
        mbobs_list: list of MultiBandObsList
            One for each object.  If it is a simple MultiBandObsList it will
            be converted to a list

        Returns
        -------
        data: np.ndarray
            Array with all output fields.
        epochs_data : np.ndarray
            Array of data about the epochs.
        """
        if not isinstance(mbobs_list, list):
            mbobs_list = [mbobs_list]

        mofc = self['mof']
        lm_pars = mofc.get('lm_pars', None)

        try:
            _fit_all_psfs(mbobs_list, self['mof']['psf'])
            _measure_all_psf_fluxes(mbobs_list)

            epochs_data = self._get_epochs_output(mbobs_list)

            fitter = self._mof_fitter_class(
                mbobs_list,
                mofc['model'],
                prior=self.mof_prior,
                lm_pars=lm_pars,
            )
            for i in range(ntry):
                logger.debug('try: %d' % (i+1))
                guess = self._guess_func(
                    mbobs_list,
                    mofc['detband'],
                    mofc['model'],
                    self.rng,
                    prior=self.mof_prior,
                )
                fitter.go(guess)

                res = fitter.get_result()
                if res['flags'] == 0:
                    break

            res['ntry'] = i+1

            if res['flags'] != 0:
                res['main_flags'] = procflags.OBJ_FAILURE
                res['main_flagstr'] = procflags.get_flagname(res['main_flags'])
            else:
                res['main_flags'] = 0
                res['main_flagstr'] = procflags.get_flagname(0)

        except NoDataError as err:
            epochs_data = None
            print(str(err))
            res = {
                'ntry': -1,
                'main_flags': procflags.NO_DATA,
                'main_flagstr': procflags.get_flagname(procflags.NO_DATA),
            }

        except BootPSFFailure as err:
            fitter = None
            epochs_data = None
            print(str(err))
            res = {
                'ntry': -1,
                'main_flags': procflags.PSF_FAILURE,
                'main_flagstr': procflags.get_flagname(procflags.PSF_FAILURE),
            }

        if res['main_flags'] != 0:
            reslist = None
        else:
            reslist = fitter.get_result_list()

        data = self._get_output(
            mbobs_list,
            res,
            reslist,
        )

        self._mof_fitter = fitter

        return data, epochs_data

    def get_mof_fitter(self):
        """
        get the MOF fitter
        """
        return self._mof_fitter

    def _set_mof_fitter_class(self):
        self._mof_fitter_class = mof.MOFStamps

    def _set_guess_func(self):
        self._guess_func = get_stamp_guesses

    def _setup(self):
        """
        set some useful values
        """
        self.npars = self.get_npars()
        self.npars_psf = self.get_npars_psf()

    @property
    def model(self):
        """
        model for fitting
        """
        return self['mof']['model']

    def get_npars(self):
        """
        number of pars we expect
        """
        return ngmix.gmix.get_model_npars(self.model) + self.nband-1

    def get_npars_psf(self):
        model = self['mof']['psf']['model']
        return 6*ngmix.gmix.get_model_ngauss(model)

    @property
    def namer(self):
        return Namer(front=self['mof']['model'])

    def _get_epochs_dtype(self):
        dt = [
            ('id', 'i8'),
            ('band', 'i2'),
            ('file_id', 'i4'),
            ('psf_pars', 'f8', self.npars_psf),
        ]
        return dt

    def _get_epochs_struct(self):
        dt = self._get_epochs_dtype()
        data = np.zeros(1, dtype=dt)
        data['id'] = -9999
        data['band'] = -1
        data['file_id'] = -1
        data['psf_pars'] = -9999
        return data

    def _get_epochs_output(self, mbobs_list):
        elist = []
        for mbobs in mbobs_list:
            for band, obslist in enumerate(mbobs):
                for obs in obslist:
                    meta = obs.meta
                    edata = self._get_epochs_struct()
                    edata['id'] = meta['id']
                    edata['band'] = band
                    edata['file_id'] = meta['file_id']
                    psf_gmix = obs.psf.gmix
                    edata['psf_pars'][0] = psf_gmix.get_full_pars()

                    elist.append(edata)

        edata = eu.numpy_util.combine_arrlist(elist)
        return edata

    def _get_dtype(self):
        npars = self.npars
        nband = self.nband

        n = self.namer
        dt = [
            ('id', 'i8'),
            ('fofid', 'i8'),  # fof id within image
            ('flags', 'i4'),
            ('flagstr', 'S18'),
            ('masked_frac', 'f4'),
            ('psf_g', 'f8', 2),
            ('psf_T', 'f8'),
            ('psf_flux_flags', 'i4', nband),
            ('psf_flux', 'f8', nband),
            ('psf_mag', 'f8', nband),
            ('psf_flux_err', 'f8', nband),
            ('psf_flux_s2n', 'f8', nband),
            (n('flags'), 'i4'),
            (n('ntry'), 'i2'),
            (n('nfev'), 'i4'),
            (n('s2n'), 'f8'),
            (n('pars'), 'f8', npars),
            (n('pars_err'), 'f8', npars),
            (n('pars_cov'), 'f8', (npars, npars)),
            (n('g'), 'f8', 2),
            (n('g_cov'), 'f8', (2, 2)),
            (n('T'), 'f8'),
            (n('T_err'), 'f8'),
            (n('T_ratio'), 'f8'),
            (n('flux'), 'f8', nband),
            (n('mag'), 'f8', nband),
            (n('flux_cov'), 'f8', (nband, nband)),
            (n('flux_err'), 'f8', nband),
        ]

        if 'bd' in self['mof']['model']:
            dt += [
                (n('fracdev'), 'f8'),
                (n('fracdev_err'), 'f8'),
            ]
        if self['mof']['model'] == 'bd':
            dt += [
                (n('logTratio'), 'f8'),
                (n('logTratio_err'), 'f8'),
            ]

        return dt

    def _get_struct(self, nobj):
        dt = self._get_dtype()
        st = np.zeros(nobj, dtype=dt)
        st['flags'] = procflags.NO_ATTEMPT
        st['flagstr'] = procflags.get_flagname(procflags.NO_ATTEMPT)

        n = self.namer
        st[n('flags')] = st['flags']

        noset = ['id', 'ra', 'dec', 'fofid',
                 'flags', 'flagstr', n('flags')]

        for n in st.dtype.names:
            if n not in noset:
                if 'err' in n or 'cof' in n:
                    st[n] = 9.999e9
                else:
                    st[n] = -9.999e9

        return st

    def _get_output(self, mbobs_list, main_res, reslist):

        nband = self.nband
        nobj = len(mbobs_list)
        output = self._get_struct(nobj)

        output['flags'] = main_res['main_flags']
        output['flagstr'] = main_res['main_flagstr']

        n = self.namer
        pn = Namer(front='psf')

        if 'flags' in main_res:
            output[n('flags')] = main_res['flags']

        output[n('ntry')] = main_res['ntry']
        logger.info('ntry: %d' % main_res['ntry'])

        # model flags will remain at NO_ATTEMPT
        if main_res['main_flags'] == 0:

            for i, res in enumerate(reslist):
                t = output[i]
                mbobs = mbobs_list[i]

                t['id'] = mbobs.meta['id']
                t['fofid'] = mbobs.meta['fofid']
                t['masked_frac'] = mbobs.meta['masked_frac']

                for band, obslist in enumerate(mbobs):
                    meta = obslist.meta

                    if nband > 1:
                        t['psf_flux_flags'][band] = meta['psf_flux_flags']
                        for name in ('flux', 'flux_err', 'flux_s2n'):
                            t[pn(name)][band] = meta[pn(name)]

                        tflux = t[pn('flux')][band].clip(min=0.001)
                        t[pn('mag')][band] = (
                            meta['magzp_ref'] - 2.5*np.log10(tflux))

                    else:
                        t['psf_flux_flags'] = meta['psf_flux_flags']
                        for name in ('flux', 'flux_err', 'flux_s2n'):
                            t[pn(name)] = meta[pn(name)]

                        tflux = t[pn('flux')].clip(min=0.001)
                        t[pn('mag')] = meta['magzp_ref'] - 2.5*np.log10(tflux)

                for name, val in res.items():
                    if name == 'nband':
                        continue

                    if 'psf' in name:
                        t[name] = val
                    else:
                        nname = n(name)
                        t[nname] = val

                        if 'pars_cov' in name:
                            ename = n('pars_err')
                            pars_err = np.sqrt(np.diag(val))
                            t[ename] = pars_err

                for band, obslist in enumerate(mbobs):
                    meta = obslist.meta
                    if nband > 1:
                        tflux = t[n('flux')][band].clip(min=0.001)
                        t[n('mag')][band] = (
                            meta['magzp_ref'] - 2.5*np.log10(tflux))
                    else:
                        tflux = t[n('flux')].clip(min=0.001)
                        t[n('mag')] = meta['magzp_ref']-2.5*np.log10(tflux)

                try:
                    pstr = ' '.join(['%8.3g' % el for el in t[n('pars')]])
                    estr = ' '.join(['%8.3g' % el for el in t[n('pars_err')]])
                except TypeError:
                    pstr = '%8.3g' % t[n('pars')]
                    estr = '%8.3g' % t[n('pars_err')]
                # logger.debug('%d pars: %s' % (i, str(t[n('pars')])))
                logger.debug('%d pars: %s' % (i, pstr))
                logger.debug('%d perr: %s' % (i, estr))

        return output
