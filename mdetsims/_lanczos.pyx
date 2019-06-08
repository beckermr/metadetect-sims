# cython: language_level=3

cimport cython
from libc.math cimport sin, M_PI, floor
import numpy as np

DTYPE = np.float64

cdef double sinc(double x) nogil:
    cdef double pi_x
    if x == 0:
        return 1
    else:
        pi_x = x * M_PI
        return sin(pi_x) / pi_x


@cython.boundscheck(False)
@cython.wraparound(False)
def lanczos_one(double[:, ::1] im1, double[::1] rows, double[::1] cols, int a):
    cdef double x, y, val1, dy, sy, dx, sx, kernel
    cdef int nx, ny, outsize, i, y_pix, x_pix
    cdef int x_s, x_f, y_s, y_f

    ny = im1.shape[0]
    nx = im1.shape[1]
    outsize = rows.shape[0]

    res1 = np.zeros(outsize, dtype=DTYPE)
    cdef double[::1] res1_view = res1
    edge = np.zeros(outsize, dtype=DTYPE)
    cdef double[::1] edge_view = edge

    with nogil:
        for i in range(outsize):
            y = rows[i]
            x = cols[i]

            # get range for kernel
            x_s = <int>(floor(x)) - a + 1
            x_f = <int>(floor(x)) + a
            y_s = <int>(floor(y)) - a + 1
            y_f = <int>(floor(y)) + a

            if (x_f < 0 or
                    x_s > nx-1 or
                    y_f < 0 or
                    y_s > ny-1):
                edge_view[i] = 1

            # now sum over the cells in the kernel
            val1 = 0.0
            for y_pix in range(y_s, y_f+1):
                if y_pix < 0 or y_pix > ny-1:
                    continue

                dy = y - y_pix
                sy = sinc(dy) * sinc(dy/a)

                for x_pix in range(x_s, x_f+1):
                    if x_pix < 0 or x_pix > nx-1:
                        continue

                    dx = x - x_pix
                    sx = sinc(dx) * sinc(dx/a)

                    kernel = sx*sy

                    val1 += im1[y_pix, x_pix] * kernel

            res1_view[i] = val1

    return res1, edge


@cython.boundscheck(False)
@cython.wraparound(False)
def lanczos_three(double[:, ::1] im1, double[:, ::1] im2, double[:, ::1] im3, double[::1] rows, double[::1] cols, int a):
    cdef double x, y, val1, val2, val3, dy, sy, dx, sx, kernel
    cdef int nx, ny, outsize, i, y_pix, x_pix
    cdef int x_s, x_f, y_s, y_f

    ny = im1.shape[0]
    nx = im1.shape[1]
    outsize = rows.shape[0]

    res1 = np.zeros(outsize, dtype=DTYPE)
    cdef double[::1] res1_view = res1
    res2 = np.zeros(outsize, dtype=DTYPE)
    cdef double[::1] res2_view = res2
    res3 = np.zeros(outsize, dtype=DTYPE)
    cdef double[::1] res3_view = res3

    edge = np.zeros(outsize, dtype=DTYPE)
    cdef double[::1] edge_view = edge

    with nogil:
        for i in range(outsize):
            y = rows[i]
            x = cols[i]

            # get range for kernel
            x_s = <int>(floor(x)) - a + 1
            x_f = <int>(floor(x)) + a
            y_s = <int>(floor(y)) - a + 1
            y_f = <int>(floor(y)) + a

            if (x_f < 0 or
                    x_s > nx-1 or
                    y_f < 0 or
                    y_s > ny-1):
                edge_view[i] = 1

            # now sum over the cells in the kernel
            val1 = 0.0
            val2 = 0.0
            val3 = 0.0
            for y_pix in range(y_s, y_f+1):
                if y_pix < 0 or y_pix > ny-1:
                    continue

                dy = y - y_pix
                sy = sinc(dy) * sinc(dy/a)

                for x_pix in range(x_s, x_f+1):
                    if x_pix < 0 or x_pix > nx-1:
                        continue

                    dx = x - x_pix
                    sx = sinc(dx) * sinc(dx/a)

                    kernel = sx*sy

                    val1 += im1[y_pix, x_pix] * kernel
                    val2 += im2[y_pix, x_pix] * kernel
                    val3 += im3[y_pix, x_pix] * kernel

            res1_view[i] = val1
            res2_view[i] = val2
            res3_view[i] = val3

    return res1, res2, res3, edge
