import numpy as np

from ._lanczos import lanczos_one, lanczos_three


def lanczos_resample_one(im1, rows, cols, a=3):
    """Lanczos resample one image at the input row and column positions.

    Points whose interpolation kernel would be truncated because it extends
    beyond the input image are marked with edge=True in the output
    edge boolean array

    Parameters
    ----------
    im1 : np.ndarray
        A two-dimensional array with the image values.
    rows : np.ndarray
        A one-dimensional array of input row/y values. These denote the
        location to sample on the first, slowest moving axis of the image.
    cols : np.ndarray
        A one-dimensional array of input column/x values. These denote the
        location to sample on the second, fastest moving axis of the image.
    a : int, optional
        The size of the Lanczos kernel. The default of 3 is a good choice
        for many applications.

    Returns
    -------
    values1, edge : np.ndarray
        The resampled value for each row, column pair. Points whose
        interpolation kernal was truncated because it extended beyond
        the input image have edge=True
    """

    if im1 is None or rows is None or cols is None:
        raise ValueError('lanczos_resample_one cannot handle None inputs!')

    rows = np.atleast_1d(rows)
    cols = np.atleast_1d(cols)

    if np.ndim(rows) != 1 or np.ndim(cols) != 1:
        raise ValueError('rows, cols must be 1-d!')

    values1, edge = lanczos_one(
        np.require(im1, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
        np.require(rows, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
        np.require(cols, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
        int(a))

    return values1, edge.astype(np.bool)


def lanczos_resample_three(im1, im2, im3, rows, cols, a=3):
    """Lanczos resample three images at the input row and column positions.

    Points whose interpolation kernel would be truncated because it extends
    beyond the input image are marked with edge=True in the output
    edge boolean array

    Parameters
    ----------
    im1 : np.ndarray
        A two-dimensional array with the image values.
    im2 : np.ndarray
        A two-dimensional array with the image values.
    im3 : np.ndarray
        A two-dimensional array with the image values.
    rows : np.ndarray
        A one-dimensional array of input row/y values. These denote the
        location to sample on the first, slowest moving axis of the image.
    cols : np.ndarray
        A one-dimensional array of input column/x values. These denote the
        location to sample on the second, fastest moving axis of the image.
    a : int, optional
        The size of the Lanczos kernel. The default of 3 is a good choice
        for many applications.

    Returns
    -------
    values1, values2, values3, edge : np.ndarray
        The resampled value for each row, column pair. Points whose
        interpolation kernal was truncated because it extended beyond
        the input image have edge=True
    """

    if (im1 is None or im2 is None or im3 is None or
            rows is None or cols is None):
        raise ValueError('lanczos_resample_one cannot handle None inputs!')

    rows = np.atleast_1d(rows)
    cols = np.atleast_1d(cols)

    if np.ndim(rows) != 1 or np.ndim(cols) != 1:
        raise ValueError('rows, cols must be 1-d!')

    values1, values2, values3, edge = lanczos_three(
        np.require(im1, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
        np.require(im2, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
        np.require(im3, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
        np.require(rows, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
        np.require(cols, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
        int(a))

    return values1, values2, values3, edge.astype(np.bool)
