import copy
import numpy as np


def get_fofs(cat, fof_conf, mask=None):
    """Generate FoF groups.

    Parameters
    ----------
    cat: array with fields
        Usually the cat from a meds file
    fof_conf: dict
        configuration for the FoF group finder
    mask: mask object
        e.g. a star mask.  Objects that are masked are put into
        their own FoF group, and are not allowed to be part
        of a group with other objects
    """

    if mask is not None:
        is_masked = mask.is_masked(cat['ra'], cat['dec'])
    else:
        is_masked = None

    mn = MEDSNbrs(
        cat,
        fof_conf,
        is_masked=is_masked,
    )

    nbr_data = mn.get_nbrs()

    nf = NbrsFoF(nbr_data)
    fofs = nf.get_fofs()

    return nbr_data, fofs


def make_singleton_fofs(cat):
    """Generate a fofs file, one object per groups

    Parameters
    ----------
    cat: array with fields
        Should have a 'number' entry

    Returns
    -------
    Fof group array with fields, entries 'fofid', 'number'
    """
    dt = [('fofid', 'i8'), ('number', 'i8')]
    fofs = np.zeros(cat.size, dtype=dt)
    fofs['fofid'] = np.arange(fofs.size)
    fofs['number'] = cat['number']
    return fofs


class MEDSNbrs(object):
    """Gets nbrs of any postage stamp in the MEDS.

    A nbr is defined as any stamp which overlaps the stamp under consideration
    given a buffer or is in the seg map. See the code below.

    Options:
        buff_type - how to compute buffer length for stamp overlap
            'min': minimum of two stamps
            'max': max of two stamps
            'tot': sum of two stamps

        buff_frac - fraction by whch to multiply the buffer

        maxsize_to_replace - postage stamp size to replace with maxsize
        maxsize - size ot use instead of maxsize_to_replace to compute overlap

        check_seg - use object's seg map to get nbrs in addition to postage
                    stamp overlap
    """
    def __init__(self, meds, conf, is_masked=None):
        self.meds = meds
        self.conf = conf

        if is_masked is None:
            is_masked = np.zeros(meds.size, dtype=bool)
        self.is_masked = is_masked
        self.is_unmasked = ~is_masked

        self._init_bounds()

    def _init_bounds(self):
        if self.conf['method'] == 'radius':
            return self._init_bounds_by_radius()
        else:
            raise NotImplementedError(
                'stamps not implemented for ra,dec version')
            return self._init_bounds_by_stamps()

    def _init_bounds_by_radius(self):

        radius_name = self.conf['radius_column']

        min_radius = self.conf.get('min_radius_arcsec', None)
        if min_radius is None:
            # arcsec
            min_radius = 1.0

        max_radius = self.conf.get('max_radius_arcsec', None)
        if max_radius is None:
            max_radius = np.inf

        m = self.meds

        # switch to row, col MRB
        # med_ra = np.median(m['ra'])
        # med_dec = np.median(m['dec'])

        # first get median pixel scale
        _pixel_scale = np.median(np.sqrt(
            m['dvdrow'][:, 0] * m['dudcol'][:, 0] -
            m['dvdcol'][:, 0] * m['dudrow'][:, 0]))

        # now make fake ra-dec using the median pixel scale
        _ra = m['orig_row'][:, 0] * _pixel_scale
        _dec = m['orig_col'][:, 0] * _pixel_scale
        med_ra = np.median(_ra)
        med_dec = np.median(_dec)

        r = m[radius_name].copy()

        r *= self.conf['radius_mult']

        r.clip(min=min_radius, max=max_radius, out=r)

        r += self.conf['padding_arcsec']

        # factor of 2 because this should be a diameter as it is used later
        diameter = r*2
        self.sze = diameter

        ra_diff = (_ra - med_ra)
        dec_diff = (_dec - med_dec)

        self.l = ra_diff - r  # noqa
        self.r = ra_diff + r
        self.b = dec_diff - r
        self.t = dec_diff + r

    def get_nbrs(self, verbose=True):
        nbrs_data = []
        dtype = [('number', 'i8'), ('nbr_number', 'i8')]

        for mindex in range(self.meds.size):
            nbrs = self.check_mindex(mindex)

            # add to final list
            for nbr in nbrs:
                nbrs_data.append((self.meds['number'][mindex], nbr))

        # return array sorted by number
        nbrs_data = np.array(nbrs_data, dtype=dtype)
        i = np.argsort(nbrs_data['number'])
        nbrs_data = nbrs_data[i]

        return nbrs_data

    def check_mindex(self, mindex):
        m = self.meds

        # check that current gal has OK stamp, or return bad crap
        if (m['orig_start_row'][mindex, 0] == -9999
                or m['orig_start_col'][mindex, 0] == -9999
                or self.is_masked[mindex]):

            nbr_numbers = np.array([-1], dtype=int)
            return nbr_numbers

        q, = np.where(
            (self.l[mindex] < self.r)
            &
            (self.r[mindex] > self.l)
        )
        if q.size > 0:
            qt, = np.where(
                (self.t[mindex] > self.b[q])
                &
                (self.b[mindex] < self.t[q])
            )
            q = q[qt]
            if q.size > 0:
                # remove dups and crap
                qt, = np.where(
                    (m['number'][mindex] != m['number'][q])
                    &
                    (m['orig_start_row'][q, 0] != -9999)
                    &
                    (m['orig_start_col'][q, 0] != -9999)
                )
                q = q[qt]

        nbr_numbers = m['number'][q]
        if nbr_numbers.size > 0:
            nbr_numbers = np.unique(nbr_numbers)
            inds = nbr_numbers-1
            q, = np.where(
                (m['orig_start_row'][inds, 0] != -9999) &
                (m['orig_start_col'][inds, 0] != -9999) &
                (self.is_unmasked[inds])
            )
            nbr_numbers = nbr_numbers[q]

        # if have stuff return unique else return -1
        if nbr_numbers.size == 0:
            nbr_numbers = np.array([-1], dtype=int)

        return nbr_numbers


class NbrsFoF(object):
    def __init__(self, nbrs_data):
        self.nbrs_data = nbrs_data
        self.Nobj = len(np.unique(nbrs_data['number']))

        # records fofid of entry
        self.linked = np.zeros(self.Nobj, dtype='i8')
        self.fofs = {}

        self._fof_data = None

    def get_fofs(self, verbose=True):
        self._make_fofs(verbose=verbose)
        return self._fof_data

    def _make_fofs(self, verbose=True):
        # init
        self._init_fofs()

        for i in range(self.Nobj):
            self._link_fof(i)

        for fofid, k in enumerate(self.fofs):
            inds = np.array(list(self.fofs[k]), dtype=int)
            self.linked[inds[:]] = fofid
        self.fofs = {}

        self._make_fof_data()

    def _link_fof(self, mind):
        # get nbrs for this object
        nbrs = set(self._get_nbrs_index(mind))

        # always make a base fof
        if self.linked[mind] == -1:
            fofid = copy.copy(mind)
            self.fofs[fofid] = set([mind])
            self.linked[mind] = fofid
        else:
            fofid = copy.copy(self.linked[mind])

        # loop through nbrs
        for nbr in nbrs:
            if self.linked[nbr] == -1 or self.linked[nbr] == fofid:
                # not linked so add to current
                self.fofs[fofid].add(nbr)
                self.linked[nbr] = fofid
            else:
                # join!
                self.fofs[self.linked[nbr]] |= self.fofs[fofid]
                del self.fofs[fofid]
                fofid = copy.copy(self.linked[nbr])
                inds = np.array(list(self.fofs[fofid]), dtype=int)
                self.linked[inds[:]] = fofid

    def _make_fof_data(self):
        self._fof_data = []
        for i in range(self.Nobj):
            self._fof_data.append((self.linked[i], i+1))
        self._fof_data = np.array(
            self._fof_data, dtype=[('fofid', 'i8'), ('number', 'i8')])
        i = np.argsort(self._fof_data['number'])
        self._fof_data = self._fof_data[i]
        assert np.all(self._fof_data['fofid'] >= 0)

    def _init_fofs(self):
        self.linked[:] = -1
        self.fofs = {}

    def _get_nbrs_index(self, mind):
        q, = np.where(
            (self.nbrs_data['number'] == mind+1) &
            (self.nbrs_data['nbr_number'] > 0))
        if len(q) > 0:
            return list(self.nbrs_data['nbr_number'][q]-1)
        else:
            return []
