import numpy as np
import esutil as eu

from metadetect.detect import MEDSifier
from .util import get_masked_frac
from .fofs import get_fofs
from .mof_fitter import MOFFitter
from .metacal_fitter import MetacalFitter, METACAL_TYPES


class MetacalPlusMOF(object):
    """Run metacal+MOF on an image by running source detection, making
    FoFs, fitting MOF models, and then running metacal.

    Parameters
    ----------
    conf : dict
        A configuration dictionary.
        See `mdetsims.defaults.TEST_METACAL_MOF_CONFIG` for details.
    mbobs : ngmix.MultibandObsList
        A single epoch, possibly multiband image to measure.
    rng : np.random.RandomState
        An RNG instance to use.

    Methods
    -------
    go()
        Run the steps above and set the result.

    Attributes
    ----------
    result : dict
        A python dictionary with the results for each shear ('noshear', '1p',
        '1m', '2p', '2m') and the multi-object fitting ('mof').
    """
    def __init__(self, conf, mbobs, rng):
        self.conf = conf
        self.mbobs = mbobs
        self.rng = rng

    @property
    def result(self):
        """The fitting results."""
        if not hasattr(self, '_result'):
            raise RuntimeError('run go() first')

        return self._result

    def go(self):
        """Run the fitting steps and set the result."""

        # first run detection
        self._medsifier = MEDSifier(
            mbobs=self.mbobs,
            sx_config=self.conf['sx'],
            meds_config=self.conf['meds'])
        self._mbmeds = self._medsifier.get_multiband_meds()

        # then make the fofs
        nbr_data, fofs = get_fofs(self._medsifier.cat, self.conf['fofs'])
        self._nbr_data = nbr_data
        self._fofs = fofs

        # now for each fof, assemble the observations and run the code
        data = {mtype: [] for mtype in METACAL_TYPES}

        n_fofs = np.max(self._fofs['fofid'])
        for fofid in range(n_fofs):
            msk = self._fofs['fofid'] == fofid
            inds = np.where(msk)[0]

            list_of_mbobs = []
            for ind in inds:
                o = self._mbmeds.get_mbobs(
                    ind, weight_type=self.conf['weight_type'])
                o.meta['id'] = ind
                o.meta['fofid'] = fofid
                o.meta['masked_frac'] = get_masked_frac(o)
                for i in range(len(o)):
                    # these settings do not matter that much I think
                    o[i].meta['Tsky'] = 1
                    o[i].meta['magzp_ref'] = 26.5
                list_of_mbobs.append(o)

            nband = len(list_of_mbobs[0])
            mcal = MetacalFitter(
                self.conf, nband, self.rng,
                mof_fitter=MOFFitter(self.conf, nband, self.rng))

            mcal.go(list_of_mbobs)
            res = mcal.result

            if res is not None:
                for key in res:
                    if res[key] is not None:
                        data[key].append(res[key])

        for key in data:
            if len(data[key]) > 0:
                # combine, sort by ind
                res = eu.numpy_util.combine_arrlist(data[key])
                srt = np.argsort(res['id'])
                res = res[srt]
                data[key] = res
            else:
                data[key] = None

        self._result = data
