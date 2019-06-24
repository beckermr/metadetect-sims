import numpy as np
import esutil as eu

from ngmix.medsreaders import MultiBandNGMixMEDS
from metadetect.detect import MEDSInterface
from .metacal_fitter import MetacalFitter, METACAL_TYPES


class TruthMEDSifier(object):
    def __init__(self,
                 mbobs,
                 meds_config,
                 truth_cat):
        """
        very simple MEDS maker for images. Assumes the images are perfectly
        registered and are sky subtracted, with constant PSF and WCS.

        The images are added together to make a detection image and sep, the
        SExtractor wrapper, is run

        parameters
        ----------
        mbobs: ngmix.MultiBandObsList
            The data
        meds_config: dict, optional
            Dict holding MEDS parameters
        cat : np.recarray
            The truth catalog.
        """
        self.mbobs = mbobs
        self.nband = len(mbobs)
        assert len(mbobs[0]) == 1, 'multi-epoch is not supported'

        self.meds_config = meds_config
        self.truth_cat = truth_cat

        self._set_cat_and_seg()

    def get_multiband_meds(self):
        """
        get a MultiBandMEDS object holding all bands
        """

        mlist = []
        for band in range(self.nband):
            m = self.get_meds(band)
            mlist.append(m)

        return MultiBandNGMixMEDS(mlist)

    def get_meds(self, band):
        """
        get fake MEDS interface to the specified band
        """
        obslist = self.mbobs[band]
        obs = obslist[0]
        return MEDSInterface(
            obs,
            self.seg,
            self.cat,
        )

    def _set_cat_and_seg(self):
        ncut = 2  # need this to make sure array
        new_dt = [
            ('id', 'i8'),
            ('number', 'i4'),
            ('ncutout', 'i4'),
            ('box_size', 'i4'),
            ('file_id', 'i8', ncut),
            ('orig_row', 'f4', ncut),
            ('orig_col', 'f4', ncut),
            ('orig_start_row', 'i8', ncut),
            ('orig_start_col', 'i8', ncut),
            ('orig_end_row', 'i8', ncut),
            ('orig_end_col', 'i8', ncut),
            ('cutout_row', 'f4', ncut),
            ('cutout_col', 'f4', ncut),
            ('dudrow', 'f8', ncut),
            ('dudcol', 'f8', ncut),
            ('dvdrow', 'f8', ncut),
            ('dvdcol', 'f8', ncut),
        ]
        cat = eu.numpy_util.add_fields(self.truth_cat, new_dt)
        cat['id'] = np.arange(cat.size)
        cat['number'] = np.arange(1, cat.size+1)
        cat['ncutout'] = 1

        jacob = self.mbobs[0][0].jacobian
        cat['dudrow'][:, 0] = jacob.dudrow
        cat['dudcol'][:, 0] = jacob.dudcol
        cat['dvdrow'][:, 0] = jacob.dvdrow
        cat['dvdcol'][:, 0] = jacob.dvdcol

        box_size = (
            np.ones(cat.size, dtype=np.int32) *
            self.meds_config['min_box_size'])

        half_box_size = box_size//2

        maxrow, maxcol = self.mbobs[0][0].image.shape

        cat['box_size'] = box_size

        cat['orig_row'][:, 0] = cat['y']
        cat['orig_col'][:, 0] = cat['x']

        orow = cat['orig_row'][:, 0].astype('i4')
        ocol = cat['orig_col'][:, 0].astype('i4')

        ostart_row = orow - half_box_size + 1
        ostart_col = ocol - half_box_size + 1
        oend_row = orow + half_box_size + 1  # plus one for slices
        oend_col = ocol + half_box_size + 1

        ostart_row.clip(min=0, out=ostart_row)
        ostart_col.clip(min=0, out=ostart_col)
        oend_row.clip(max=maxrow, out=oend_row)
        oend_col.clip(max=maxcol, out=oend_col)

        # could result in smaller than box_size above
        cat['orig_start_row'][:, 0] = ostart_row
        cat['orig_start_col'][:, 0] = ostart_col
        cat['orig_end_row'][:, 0] = oend_row
        cat['orig_end_col'][:, 0] = oend_col
        cat['cutout_row'][:, 0] = \
            cat['orig_row'][:, 0] - cat['orig_start_row'][:, 0]
        cat['cutout_col'][:, 0] = \
            cat['orig_col'][:, 0] - cat['orig_start_col'][:, 0]

        self.seg = np.zeros_like(self.mbobs[0][0].image, dtype=np.int32)
        self.cat = cat


class MetacalTrueDetect(object):
    """Run metacal on an image using the true source locations and then
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
    cat : np.recarray
        The true source catalog. Must have y and x positions set.

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
    def __init__(self, conf, mbobs, rng, cat):
        self.conf = conf
        self.mbobs = mbobs
        self.rng = rng
        self.cat = cat

    @property
    def result(self):
        """The fitting results."""
        if not hasattr(self, '_result'):
            raise RuntimeError('run go() first')

        return self._result

    def go(self):
        """Run the fitting steps and set the result."""

        self._medsifier = TruthMEDSifier(
            mbobs=self.mbobs,
            meds_config=self.conf['meds'],
            truth_cat=self.cat)
        self._mbmeds = self._medsifier.get_multiband_meds()

        # run metacal for each true object
        data = {mtype: [] for mtype in METACAL_TYPES}
        for ind in range(self.cat.size):
            o = self._mbmeds.get_mbobs(ind)
            o.meta['id'] = ind

            nband = len(o)
            mcal = MetacalFitter(self.conf, nband, self.rng)

            mcal.go([o])
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
