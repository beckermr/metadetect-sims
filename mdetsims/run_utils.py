import numpy as np
import tqdm

from .metacal import METACAL_TYPES


def cut_nones(presults, mresults):
    """Cut entries that are None in a pair of lists. Any entry that is None
    in either list will exclude the item in the other.

    Parameters
    ----------
    presults : list
        One the list of things.
    mresults : list
        The other list of things.

    Returns
    -------
    pcut : list
        The cut list.
    mcut : list
        The cut list.
    """
    prr_keep = []
    mrr_keep = []
    for pr, mr in zip(presults, mresults):
        if pr is None or mr is None:
            continue
        prr_keep.append(pr)
        mrr_keep.append(mr)

    return prr_keep, mrr_keep


def estimate_m_and_c(presults, mresults, g_true, swap12=False, step=0.01):
    """Estimate m and c from paired lensing simulations.

    Parameters
    ----------
    presults : list of iterables
        A list of iterables, each with g1p, g1m, g1, g2p, g2m, g2
        from running metadetect with a `g1` shear in the 1-component and
        0 true shear in the 2-component.
    mresults : list of iterables
        A list of iterables, each with g1p, g1m, g1, g2p, g2m, g2
        from running metadetect with a -`g1` shear in the 1-component and
        0 true shear in the 2-component.
    g_true : float
        The true value of the shear on the 1-axis in the simulation. The other
        axis is assumd to havea true value of zero.
    swap12 : bool, optional
        If True, swap the roles of the 1- and 2-axes in the computation.
    step : float, optional
        The step used in metadetect for estimating the response. Default is
        0.01.

    Returns
    -------
    m : float
        Estimate of the multiplicative bias.
    merr : float
        Estimat of the 1-sigma standard error in `m`.
    c : float
        Estimate of the additive bias.
    cerr : float
        Estimate of the 1-sigma standard error in `c`.
    """

    prr_keep, mrr_keep = cut_nones(presults, mresults)

    def _get_stuff(rr):
        _a = np.vstack(rr)
        g1p = _a[:, 0]
        g1m = _a[:, 1]
        g1 = _a[:, 2]
        g2p = _a[:, 3]
        g2m = _a[:, 4]
        g2 = _a[:, 5]

        if swap12:
            g1p, g1m, g1, g2p, g2m, g2 = g2p, g2m, g2, g1p, g1m, g1

        return (
            g1, (g1p - g1m) / 2 / step * g_true,
            g2, (g2p - g2m) / 2 / step)

    g1p, R11p, g2p, R22p = _get_stuff(prr_keep)
    g1m, R11m, g2m, R22m = _get_stuff(mrr_keep)

    msk = (
        np.isfinite(g1p) &
        np.isfinite(R11p) &
        np.isfinite(g1m) &
        np.isfinite(R11m) &
        np.isfinite(g2p) &
        np.isfinite(R22p) &
        np.isfinite(g2m) &
        np.isfinite(R22m))
    g1p = g1p[msk]
    R11p = R11p[msk]
    g1m = g1m[msk]
    R11m = R11m[msk]
    g2p = g2p[msk]
    R22p = R22p[msk]
    g2m = g2m[msk]
    R22m = R22m[msk]

    x1 = (R11p + R11m)/2
    y1 = (g1p - g1m) / 2

    x2 = (R22p + R22m) / 2
    y2 = (g2p + g2m) / 2

    rng = np.random.RandomState(seed=100)
    mvals = []
    cvals = []
    for _ in tqdm.trange(500, leave=False):
        ind = rng.choice(len(y1), replace=True, size=len(y1))
        mvals.append(np.mean(y1[ind]) / np.mean(x1[ind]) - 1)
        cvals.append(np.mean(y2[ind]) / np.mean(x2[ind]))

    return (
        np.mean(y1) / np.mean(x1) - 1, np.std(mvals),
        np.mean(y2) / np.mean(x2), np.std(cvals))


def measure_shear_metadetect(res, *, s2n_cut, t_ratio_cut, cut_interp):
    """Measure the shear parameters for metadetect.

    NOTE: Returns None if nothing can be measured.

    Parameters
    ----------
    res : dict
        The metadetect results.
    s2n_cut : float
        The cut on `wmom_s2n`. Typically 10.
    t_ratio_cut : float
        The cut on `t_ratio_cut`. Typically 1.2.
    cut_interp : bool
        If True, cut on the `ormask` flags.

    Returns
    -------
    g1p : float
        The mean 1-component shape for the plus metadetect measurement.
    g1m : float
        The mean 1-component shape for the minus metadetect measurement.
    g1 : float
        The mean 1-component shape for the zero-shear metadetect measurement.
    g2p : float
        The mean 2-component shape for the plus metadetect measurement.
    g2m : float
        The mean 2-component shape for the minus metadetect measurement.
    g2 : float
        The mean 2-component shape for the zero-shear metadetect measurement.
    """
    def _mask(data):
        if cut_interp:
            return (
                (data['flags'] == 0) &
                (data['ormask'] == 0) &
                (data['wmom_s2n'] > s2n_cut) &
                (data['wmom_T_ratio'] > t_ratio_cut))
        else:
            return (
                (data['flags'] == 0) &
                (data['wmom_s2n'] > s2n_cut) &
                (data['wmom_T_ratio'] > t_ratio_cut))

    op = res['1p']
    q = _mask(op)
    if not np.any(q):
        return None
    g1p = op['wmom_g'][q, 0]

    om = res['1m']
    q = _mask(om)
    if not np.any(q):
        return None
    g1m = om['wmom_g'][q, 0]

    o = res['noshear']
    q = _mask(o)
    if not np.any(q):
        return None
    g1 = o['wmom_g'][q, 0]
    g2 = o['wmom_g'][q, 1]

    op = res['2p']
    q = _mask(op)
    if not np.any(q):
        return None
    g2p = op['wmom_g'][q, 1]

    om = res['2m']
    q = _mask(om)
    if not np.any(q):
        return None
    g2m = om['wmom_g'][q, 1]

    return (
        np.mean(g1p), np.mean(g1m), np.mean(g1),
        np.mean(g2p), np.mean(g2m), np.mean(g2))


def measure_shear_metacal_plus_mof(res, *, s2n_cut, t_ratio_cut):
    """Measure the shear parameters for metacal+MOF.

    NOTE: Returns None if nothing can be measured.

    Parameters
    ----------
    res : dict
        The metacal results.
    s2n_cut : float
        The cut on `wmom_s2n`. Typically 10.
    t_ratio_cut : float
        The cut on `t_ratio_cut`. Typically 0.5.

    Returns
    -------
    g1p : float
        The mean 1-component shape for the plus metacal measurement.
    g1m : float
        The mean 1-component shape for the minus metacal measurement.
    g1 : float
        The mean 1-component shape for the zero-shear metacal measurement.
    g2p : float
        The mean 2-component shape for the plus metacal measurement.
    g2m : float
        The mean 2-component shape for the minus metacal measurement.
    g2 : float
        The mean 2-component shape for the zero-shear metacal measurement.
    """
    def _mask(mof, cat, s2n_cut=10, size_cut=0.5):
        return (
            (mof['flags'] == 0) &
            (cat['mcal_s2n'] > s2n_cut) &
            (cat['mcal_T_ratio'] > t_ratio_cut))

    msks = {}
    for sh in METACAL_TYPES:
        msks[sh] = _mask(res['mof'], res[sh])
        if not np.any(msks[sh]):
            return None

    g1p = res['1p']['mcal_g'][msks['1p'], 0]
    g1m = res['1m']['mcal_g'][msks['1m'], 0]

    g2p = res['2p']['mcal_g'][msks['2p'], 1]
    g2m = res['2m']['mcal_g'][msks['2m'], 1]

    g1 = res['noshear']['mcal_g'][msks['noshear'], 0]
    g2 = res['noshear']['mcal_g'][msks['noshear'], 1]

    return (
        np.mean(g1p), np.mean(g1m), np.mean(g1),
        np.mean(g2p), np.mean(g2m), np.mean(g2))
