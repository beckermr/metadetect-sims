from ngmix.bootstrap import Bootstrapper
from ngmix import metacal
from metadetect.fitting import Moments


class MomentsMetacalBootstrapper(Bootstrapper):

    def get_metacal_result(self):
        """
        get result of metacal
        """
        if not hasattr(self, 'metacal_res'):
            raise RuntimeError("you need to run fit_metacal first")
        return self.metacal_res

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

        res = self._do_moments(obs_dict, moments_pars, rng)

        self.metacal_res = res

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

    def _do_moments(self, obs_dict, moments_pars, rng):

        # overall flags, or'ed from each moments fit
        res = {'mcal_flags': 0}
        for key in sorted(obs_dict):
            fitter = Moments(moments_pars, rng)
            fres = fitter.go([obs_dict[key]])
            res['mcal_flags'] |= fres['flags'][0]
            tres = {}
            for name in fres.dtype.names:
                no_wmom = name.replace('wmom_', '')
                tres[no_wmom] = fres[name][0]
            res[key] = tres

        return res
