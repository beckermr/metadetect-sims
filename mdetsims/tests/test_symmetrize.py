import numpy as np

from ..symmetrize import symmetrize_bad_mask


def test_symmetrize_bmask():
    bmask = np.zeros((4, 4), dtype=bool)
    bmask[:, 0] = True
    symmetrize_bad_mask(bmask)

    assert np.array_equal(
        bmask,
        [[1, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 1, 1, 1]])
