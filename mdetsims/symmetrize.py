import numpy as np


def symmetrize_bad_mask(bad_mask):
    """Symmetrize masked pixels.

    WARNING: This function operates in-place!

    Parameters
    ----------
    bad_mask : array-like
        A boolean mask of bad pixels.
    """
    if bad_mask.shape[0] != bad_mask.shape[1]:
        raise ValueError("Only square images can be symmetrized!")

    if np.any(bad_mask):
        bm_rot = np.rot90(bad_mask)
        bad_mask |= bm_rot
