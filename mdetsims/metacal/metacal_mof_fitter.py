import numpy as np
import esutil as eu

import ngmix

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
        data = []
        n_fofs = np.max(self._fofs['fofid'])
        for fofid in range(n_fofs):
            msk = self._fofs['fofid'] == fofid
            inds = np.where(msk)[0]

            list_of_mbobs = []
            for ind in inds:
                o = self._mbmeds.get_mbobs(ind)
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

            try:
                mcal.go(list_of_mbobs)
                res = mcal.result
            except ngmix.gexceptions.GMixRangeError as e:
                print(repr(e))
                res = None

            if res is not None:
                data.append(res)

        if len(data) > 0:
            # combine, sort by ind
            res = eu.numpy_util.combine_arrlist(data)
            srt = np.argsort(res['id'])
            res = res[srt]
            self._result = self._result_to_dict(res)
        else:
            self._result = None

    def _result_to_dict(self, data):
        cols_to_always_keep = ['id', 'fofid', 'masked_frac', 'x', 'y']

        def _get_col_type(col):
            for dtup in data.descr.descr:
                if dtup[0] == col:
                    return list(dtup[1:])
            return None

        result = {}

        # build the main catalog
        main_dtype_descr = []
        for dtup in data.dtype.descr:
            if not dtup[0].startswith('mcal_'):
                main_dtype_descr.append(dtup)
        main = np.zeros(len(data), dtype=main_dtype_descr)
        for col in main.dtype.names:
            main[col] = data[col]
        result['mof'] = main

        # now build each of other catalogs
        for sh in METACAL_TYPES:
            dtype_descr = []
            for dtup in data.dtype.descr:
                if dtup[0] in cols_to_always_keep:
                    dtype_descr.append(dtup)
                elif dtup[0].startswith('mcal_') and dtup[0].endswith(sh):
                    dlist = [dtup[0].replace('_%s' % sh, '')]
                    dlist = dlist + list(dtup[1:])
                    dtype_descr.append(tuple(dlist))

            sh_cat = np.zeros(len(data), dtype=dtype_descr)
            for col in sh_cat.dtype.names:
                sh_col = col + '_%s' % sh
                if col in data.dtype.names:
                    sh_cat[col] = data[col]
                    continue
                elif sh_col in data.dtype.names:
                    sh_cat[col] = data[sh_col]
                else:
                    raise ValueError("column %s not found!" % col)
            result[sh] = sh_cat

        return result
