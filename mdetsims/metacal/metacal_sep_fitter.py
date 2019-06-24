import esutil as eu

from metadetect.detect import MEDSifier
from .metacal_fitter import MetacalFitter, METACAL_TYPES


class MetacalSepDetect(object):
    """Run metacal on an image by running source detection and then
    running metacal.

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

        # now fit
        data = {mtype: [] for mtype in METACAL_TYPES}
        for ind in range(self._medsifier.cat.size):
            o = self._mbmeds.get_mbobs(ind)
            o.meta['id'] = ind
            list_of_mbobs = [o]

            nband = len(list_of_mbobs[0])
            mcal = MetacalFitter(self.conf, nband, self.rng)
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
                # mock this up
                res = eu.numpy_util.add_fields(
                    res,
                    [('flags', 'i4')])
                res['flags'] = 0
                data[key] = res
            else:
                data[key] = None

        self._result = data
