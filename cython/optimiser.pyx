# Author: Merlijn Wajer <merlijn@archive.org>
# License: AGPL-v3

import numpy as np
cimport numpy as np
cimport cython

UINT8DTYPE = np.uint8
ctypedef np.uint8_t UINT8DTYPE_t

# Speed up the code and let's make sure all variables have a c type, otherwise performs goes out of
# the window
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.warn.undeclared(True)
def optimise_gray(np.ndarray[UINT8DTYPE_t, ndim=2] mask,
        np.ndarray[UINT8DTYPE_t, ndim=2] img, int width, int height, int n_size):
    cdef np.ndarray[UINT8DTYPE_t, ndim=2] new_img
    cdef int x, y
    cdef int val_count, val, ys, ye, xs, xe, xx, yy

    # TODO: weights for distance?

    new_img = np.copy(img)
    for y in range(0, height):
        for x in range(0, width):
            if not mask[y, x]:
                val_count = 0
                val = 0

                ys = max(0, y - n_size)
                ye = min(height, y + n_size)
                xs = max(0, x - n_size)
                xe = min(width, x + n_size)

                for yy in range(ys, ye):
                    for xx in range(xs, xe):
                        if mask[yy, xx]:
                            val += img[yy, xx]
                            val_count += 1

                for yy in range(ys, y):
                    for xx in range(xs, x):
                        val += new_img[yy, xx]
                        val_count += 1

                if val_count > 0:
                    new_img[y, x] = val / val_count
                else:
                    new_img[y, x] = 0

    return new_img


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.warn.undeclared(True)
def optimise_rgb(np.ndarray[UINT8DTYPE_t, ndim=2] mask,
        np.ndarray[UINT8DTYPE_t, ndim=3] img, int width, int height, int n_size):
    cdef np.ndarray[UINT8DTYPE_t, ndim=3] new_img
    cdef int x, y
    cdef int val_count, ys, ye, xs, xe, xx, yy
    cdef int r, g, b

    # TODO: weights for distance?

    new_img = np.copy(img)
    for y in range(0, height):
        for x in range(0, width):
            if not mask[y, x]:
                val_count = 0
                r = 0
                g = 0
                b = 0

                ys = max(0, y - n_size)
                ye = min(height, y + n_size)
                xs = max(0, x - n_size)
                xe = min(width, x + n_size)

                for yy in range(ys, ye):
                    for xx in range(xs, xe):
                        if mask[yy, xx]:
                            b += img[yy, xx, 0]
                            g += img[yy, xx, 1]
                            r += img[yy, xx, 2]
                            val_count += 1

                for yy in range(ys, y):
                    for xx in range(xs, x):
                        b += new_img[yy, xx, 0]
                        g += new_img[yy, xx, 1]
                        r += new_img[yy, xx, 2]
                        val_count += 1

                if val_count > 0:
                    new_img[y, x, 0] = b / val_count
                    new_img[y, x, 1] = g / val_count
                    new_img[y, x, 2] = r / val_count
                else:
                    new_img[y, x, 0] = 0
                    new_img[y, x, 1] = 0
                    new_img[y, x, 2] = 0

    return new_img
