import numpy as np
from numba import njit


@njit
def sinc_pade(x):
    """A pseudo-Pade approximation to sinc.

    When used in lanczos interpolation, this approximation to the sinc
    function has error in the kernel of at most ~1.03019e-05 and is exactly
    zero at x = 1, 2, 3 and exactly 1 at x = 0.
    """
    x = np.abs(x)
    num = (  # noqa
        -0.166666666666666 + x * (
        -0.289176685343373 + x * (
        -0.109757669089546 + x * (
        0.0350931080575596 + x * (
        0.0229947584643336 + x * (
        -0.00089363958201935 + x * (
        -0.00162722192965722 + x * (
        -3.00689146075626e-05 + x * (
        5.13864469774294e-05 + x * (
        1.23561563382214e-06 + x * (
        -6.37392253619041e-07))))))))))) * (x-1) * (x-2) * (x-3)
    den = (  # noqa
        1 + x * (
        -0.0982732212730775 + x * (
        0.122536542608403 + x * (
        -0.0111525324680647 + x * (
        0.00724707512833019 + x * (
        -0.000584774445653404 + x * (
        0.000262528048296579 + x * (
        -1.71596576734417e-05 + x * (
        5.91945411660804e-06 + x * (
        -2.44174818579491e-07 + x * (
        6.74473938075399e-08)))))))))))
    return num / den


@njit
def lanczos_resample_one(im1, rows, cols):
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

    Returns
    -------
    values1, edge : np.ndarray
        The resampled value for each row, column pair. Points whose
        interpolation kernal was truncated because it extended beyond
        the input image have edge=True
    """
    a = 3
    ny, nx = im1.shape
    outsize = rows.shape[0]

    res1 = np.zeros(outsize, dtype=np.float64)
    edge = np.zeros(outsize, dtype=np.bool_)

    for i in range(rows.shape[0]):
        y = rows[i]
        x = cols[i]

        # get range for kernel
        x_s = int(np.floor(x)) - a + 1
        x_f = int(np.floor(x)) + a
        y_s = int(np.floor(y)) - a + 1
        y_f = int(np.floor(y)) + a

        out_of_bounds = (
            x_f < 0 or
            x_s > nx-1 or
            y_f < 0 or
            y_s > ny-1
        )

        if out_of_bounds:
            edge[i] = True

        # now sum over the cells in the kernel
        val1 = 0.0
        for y_pix in range(y_s, y_f+1):
            if y_pix < 0 or y_pix > ny-1:
                continue

            dy = y - y_pix
            sy = sinc_pade(dy) * sinc_pade(dy/a)

            for x_pix in range(x_s, x_f+1):
                if x_pix < 0 or x_pix > nx-1:
                    continue

                dx = x - x_pix
                sx = sinc_pade(dx) * sinc_pade(dx/a)

                kernel = sx*sy

                val1 += im1[y_pix, x_pix] * kernel

        res1[i] = val1

    return res1, edge


@njit
def lanczos_resample_three(im1, im2, im3, rows, cols):
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

    Returns
    -------
    values1, values2, values3, edge : np.ndarray
        The resampled value for each row, column pair. Points whose
        interpolation kernal was truncated because it extended beyond
        the input image have edge=True
    """
    a = 3
    ny, nx = im1.shape
    outsize = rows.shape[0]

    res1 = np.zeros(outsize, dtype=np.float64)
    res2 = np.zeros(outsize, dtype=np.float64)
    res3 = np.zeros(outsize, dtype=np.float64)
    edge = np.zeros(outsize, dtype=np.bool_)

    for i in range(rows.shape[0]):
        y = rows[i]
        x = cols[i]

        # get range for kernel
        x_s = int(np.floor(x)) - a + 1
        x_f = int(np.floor(x)) + a
        y_s = int(np.floor(y)) - a + 1
        y_f = int(np.floor(y)) + a

        out_of_bounds = (
            x_f < 0 or
            x_s > nx-1 or
            y_f < 0 or
            y_s > ny-1
        )

        if out_of_bounds:
            edge[i] = True

        # now sum over the cells in the kernel
        val1 = 0.0
        val2 = 0.0
        val3 = 0.0
        for y_pix in range(y_s, y_f+1):
            if y_pix < 0 or y_pix > ny-1:
                continue

            dy = y - y_pix
            sy = sinc_pade(dy) * sinc_pade(dy/a)

            for x_pix in range(x_s, x_f+1):
                if x_pix < 0 or x_pix > nx-1:
                    continue

                dx = x - x_pix
                sx = sinc_pade(dx) * sinc_pade(dx/a)

                kernel = sx*sy

                val1 += im1[y_pix, x_pix] * kernel
                val2 += im2[y_pix, x_pix] * kernel
                val3 += im3[y_pix, x_pix] * kernel

        res1[i] = val1
        res2[i] = val2
        res3[i] = val3

    return res1, res2, res3, edge


# import numpy as np
#
# from ._lanczos import lanczos_one, lanczos_three
#
#
# def lanczos_resample_one(im1, rows, cols, a=3):
#     """Lanczos resample one image at the input row and column positions.
#
#     Points whose interpolation kernel would be truncated because it extends
#     beyond the input image are marked with edge=True in the output
#     edge boolean array
#
#     Parameters
#     ----------
#     im1 : np.ndarray
#         A two-dimensional array with the image values.
#     rows : np.ndarray
#         A one-dimensional array of input row/y values. These denote the
#         location to sample on the first, slowest moving axis of the image.
#     cols : np.ndarray
#         A one-dimensional array of input column/x values. These denote the
#         location to sample on the second, fastest moving axis of the image.
#     a : int, optional
#         The size of the Lanczos kernel. The default of 3 is a good choice
#         for many applications.
#
#     Returns
#     -------
#     values1, edge : np.ndarray
#         The resampled value for each row, column pair. Points whose
#         interpolation kernal was truncated because it extended beyond
#         the input image have edge=True
#     """
#
#     if im1 is None or rows is None or cols is None:
#         raise ValueError('lanczos_resample_one cannot handle None inputs!')
#
#     rows = np.atleast_1d(rows)
#     cols = np.atleast_1d(cols)
#
#     if np.ndim(rows) != 1 or np.ndim(cols) != 1:
#         raise ValueError('rows, cols must be 1-d!')
#
#     values1, edge = lanczos_one(
#         np.require(
#             im1, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
#         np.require(
#             rows, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
#         np.require(
#             cols, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
#         int(a))
#
#     return values1, edge.astype(np.bool)
#
#
# def lanczos_resample_three(im1, im2, im3, rows, cols, a=3):
#     """Lanczos resample three images at the input row and column positions.
#
#     Points whose interpolation kernel would be truncated because it extends
#     beyond the input image are marked with edge=True in the output
#     edge boolean array
#
#     Parameters
#     ----------
#     im1 : np.ndarray
#         A two-dimensional array with the image values.
#     im2 : np.ndarray
#         A two-dimensional array with the image values.
#     im3 : np.ndarray
#         A two-dimensional array with the image values.
#     rows : np.ndarray
#         A one-dimensional array of input row/y values. These denote the
#         location to sample on the first, slowest moving axis of the image.
#     cols : np.ndarray
#         A one-dimensional array of input column/x values. These denote the
#         location to sample on the second, fastest moving axis of the image.
#     a : int, optional
#         The size of the Lanczos kernel. The default of 3 is a good choice
#         for many applications.
#
#     Returns
#     -------
#     values1, values2, values3, edge : np.ndarray
#         The resampled value for each row, column pair. Points whose
#         interpolation kernal was truncated because it extended beyond
#         the input image have edge=True
#     """
#
#     if (im1 is None or im2 is None or im3 is None or
#             rows is None or cols is None):
#         raise ValueError('lanczos_resample_one cannot handle None inputs!')
#
#     rows = np.atleast_1d(rows)
#     cols = np.atleast_1d(cols)
#
#     if np.ndim(rows) != 1 or np.ndim(cols) != 1:
#         raise ValueError('rows, cols must be 1-d!')
#
#     values1, values2, values3, edge = lanczos_three(
#         np.require(
#             im1, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
#         np.require(
#             im2, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
#         np.require(
#             im3, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
#         np.require(
#             rows, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
#         np.require(
#             cols, dtype=np.float64, requirements=['C', 'A', 'O', 'E']),
#         int(a))
#
#     return values1, values2, values3, edge.astype(np.bool)
