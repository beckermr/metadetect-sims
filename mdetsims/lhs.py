import numpy as np


def get_optimized_lhs_design(n_dims, n_samples, n_iter=200, rng=None):
    """Compute a LHS design.

    Based on the pyDOE routines w/ some changes.

    Parameters
    ----------
    n_dims : int
        The number of dimensions.
    n_samples : int
        The number of samples in the design.
    n_iter : int
        the number of optimization iterations.
    rng : np.random.RandomState or None, optional
        An RNG to use for generating test designs.

    Returns
    -------
    pts : np.ndarray, shape (n_dims, n_samples)
        The optimized design.
    """
    maxmindist = -np.inf
    msk = ~np.eye(n_samples, dtype=bool)
    for _ in range(n_iter):
        # find the least correlated of 10 designs
        _pts = _lhscorrelate(n_dims, n_samples, 10, rng=rng)
        d2 = np.sum(
            (_pts[:, np.newaxis, :] - _pts[np.newaxis, :, :])**2, axis=-1)
        _maxmindist = np.min(d2[msk])
        # take the design with the max minimum distance between points
        if _maxmindist > maxmindist:
            maxmindist = _maxmindist
            pts = _pts
    return pts


def _lhsclassic(n, samples, rng=None):
    """copy of pyDOE routine w/ seeding"""
    rng = rng or np.random.RandomState()

    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)

    # Fill points uniformly in each interval
    u = rng.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    rdpoints = np.zeros_like(u)
    for j in range(n):
        rdpoints[:, j] = u[:, j]*(b-a) + a

    # Make the random pairings
    H = np.zeros_like(rdpoints)
    for j in range(n):
        order = rng.permutation(range(samples))
        H[:, j] = rdpoints[order, j]

    return H


def _lhscorrelate(n, samples, iterations, rng=None):
    """copy of the routine from pyDOE w/ a bug fix and seeding"""
    mincorr = np.inf

    # Minimize the components correlation coefficients
    for i in range(iterations):
        # Generate a random LHS
        Hcandidate = _lhsclassic(n, samples, rng=rng)
        R = np.corrcoef(Hcandidate.T)
        _mincorr = np.max(np.abs(R-np.eye(R.shape[0])))
        if _mincorr < mincorr:
            mincorr = _mincorr
            H = Hcandidate.copy()

    return H
