import logging
import numpy as np

from ngmix.bootstrap import Bootstrapper
from ngmix import metacal
from ngmix.gexceptions import BootPSFFailure, BootGalFailure
from metadetect.fitting import Moments

logger = logging.getLogger(__name__)


# this mixin is so bad, but I don't have a better idea

class MetacalMixin(object):
    def get_metacal_result(self):
        """
        get result of metacal
        """
        if not hasattr(self, 'metacal_res'):
            raise RuntimeError("you need to run fit_metacal first")
        return self.metacal_res

    def _get_all_metacal(self, metacal_pars, **kw):

        metacal_pars = self._extract_metacal_pars(metacal_pars)
        return metacal.get_all_metacal(self.mb_obs_list, **metacal_pars)

    def _extract_metacal_pars(self, metacal_pars_in):
        """
        make sure at least the step is specified
        """
        metacal_pars = {'step': 0.01}

        if metacal_pars_in is not None:
            metacal_pars.update(metacal_pars_in)

        return metacal_pars


class MomentsMetacalBootstrapper(Bootstrapper, MetacalMixin):
    def fit_metacal(self,
                    *,
                    rng,
                    moments_pars,
                    metacal_pars,
                    **kw):
        """
        run metacalibration

        parameters
        ----------
        moments_pars : dict
            Parameters for moments fitter
        metacal_pars: dict
            Parameters for metacal
        **kw:
            extra keywords for get_all_metacal
        """

        obs_dict = self._get_all_metacal(
            metacal_pars,
            **kw
        )

        # overall flags, or'ed from each moments fit
        res = {'mcal_flags': 0}
        for key in sorted(obs_dict):
            try:
                fitter = Moments(moments_pars, rng)
                fres = fitter.go([obs_dict[key]])
            except Exception as err:
                logger.debug(str(err))
                fres = {'flags': np.ones(1, dtype=[('flags', 'i4')])}

            res['mcal_flags'] |= fres['flags'][0]
            tres = {}
            for name in fres.dtype.names:
                no_wmom = name.replace('wmom_', '')
                tres[no_wmom] = fres[name][0]
            tres['flags'] = fres['flags'][0]  # make sure this is moved over
            res[key] = tres

        self.metacal_res = res


class MaxMetacalBootstrapper(Bootstrapper, MetacalMixin):
    def fit_metacal(self,
                    psf_model,
                    gal_model,
                    pars,
                    psf_Tguess,
                    psf_fit_pars=None,
                    metacal_pars=None,
                    prior=None,
                    psf_ntry=5,
                    ntry=1,
                    guesser=None,
                    **kw):
        """
        run metacalibration

        parameters
        ----------
        psf_model: string
            model to fit for psf
        gal_model: string
            model to fit
        pars: dict
            parameters for the maximum likelihood fitter
        psf_Tguess: float
            T guess for psf
        psf_fit_pars: dict
            parameters for psf fit
        metacal_pars: dict, optional
            Parameters for metacal, default {'step':0.01}
        prior: prior on parameters, optional
            Optional prior to apply
        psf_ntry: int, optional
            Number of times to retry psf fitting, default 5
        ntry: int, optional
            Number of times to retry fitting, default 1
        **kw:
            extra keywords for get_all_metacal
        """

        obs_dict = self._get_all_metacal(
            metacal_pars,
            **kw
        )

        res = self._do_metacal_max_fits(
            obs_dict,
            psf_model, gal_model,
            pars, psf_Tguess,
            prior, psf_ntry, ntry,
            psf_fit_pars,
            guesser=guesser,
        )

        self.metacal_res = res

    def _do_metacal_max_fits(self, obs_dict, psf_model, gal_model, pars,
                             psf_Tguess, prior, psf_ntry, ntry,
                             psf_fit_pars, guesser=None):

        # overall flags, or'ed from each bootstrapper
        res = {'mcal_flags': 0}
        for key in sorted(obs_dict):
            try:
                # run a regular Bootstrapper on these observations
                boot = Bootstrapper(obs_dict[key],
                                    find_cen=self.find_cen,
                                    verbose=self.verbose)

                boot.fit_psfs(psf_model, psf_Tguess, ntry=psf_ntry,
                              fit_pars=psf_fit_pars,
                              skip_already_done=False)
                boot.fit_max(
                    gal_model,
                    pars,
                    guesser=guesser,
                    prior=prior,
                    ntry=ntry,
                )
                tres = boot.get_max_fitter().get_result()

                wsum = 0.0
                Tpsf_sum = 0.0
                gpsf_sum = np.zeros(2)
                npsf = 0
                for obslist in boot.mb_obs_list:
                    for obs in obslist:
                        if hasattr(obs, 'psf_nopix'):
                            g1, g2, T = obs.psf_nopix.gmix.get_g1g2T()
                        else:
                            g1, g2, T = obs.psf.gmix.get_g1g2T()

                        # TODO we sometimes use other weights
                        twsum = obs.weight.sum()

                        wsum += twsum
                        gpsf_sum[0] += g1*twsum
                        gpsf_sum[1] += g2*twsum
                        Tpsf_sum += T*twsum
                        npsf += 1

                tres['gpsf'] = gpsf_sum/wsum
                tres['Tpsf'] = Tpsf_sum/wsum

            except (BootPSFFailure, BootGalFailure) as err:
                logger.debug(str(err))
                tres = {'flags': 1}

            res['mcal_flags'] |= tres['flags']
            res[key] = tres

        return res
