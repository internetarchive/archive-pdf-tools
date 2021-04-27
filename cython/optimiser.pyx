# Authors:
# * Merlijn Wajer <merlijn@archive.org> - initial algorithm, slow implementation
# * Bas Weelinck <bas@weelinck.org> - fast versions
#
# License: AGPL-v3

import numpy as np
cimport numpy as np
cimport cython

INTDTYPE = np.int
UINT8DTYPE = np.uint8
ctypedef np.int_t INTDTYPE_t
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.warn.undeclared(True)
def optimise_gray2(np.ndarray[UINT8DTYPE_t, ndim=2] mask,
        np.ndarray[UINT8DTYPE_t, ndim=2] img, int width, int height, int n_size):
    cdef np.ndarray[UINT8DTYPE_t, ndim=2] new_img
    cdef int x, y
    cdef int val_count, val, ys, ye, xs, xe, xx, yy
    cdef int ifysc, ifyec, iiysc, iiyec, ifxsc, ifxec, iixsc, iixec
    cdef int inc_fir_px_val, inc_fir_px_mask

    # TODO: weights for distance?
    # This function computes a FIR and IIR version of the box blur filter incrementally
    # As seen above
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_fir_val = np.zeros([width], dtype=INTDTYPE)
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_fir_mask = np.zeros([width], dtype=INTDTYPE)

    # incremental cursors that track y-dimension FIR filter window borders
    ifysc = 0
    ifyec = 0

    # incremental cursors that track y-dimension IIR filter window borders
    iiysc = 0
    iiyec = 0

    # FIXME
    #new_img = np.empty(img.shape)
    new_img = np.copy(img)
    for y in range(0, height):
        #print('loop y')
        ys = max(0, y - n_size)
        ye = min(height, y + n_size)
        # Update y-dimension FIR window
        while ifysc < ys:
            #print('dec y')
            for x in range(0, width):
                if mask[ifysc, x]:
                    inc_fir_val[x] -= img[ifysc, x]
                    inc_fir_mask[x] -= 1
            ifysc += 1
        while ifyec < ye:
            #print('inc y')
            for x in range(0, width):
                if mask[ifyec, x]:
                    inc_fir_val[x] += img[ifyec, x]
                    inc_fir_mask[x] += 1
            ifyec += 1

        # incremental cursors that track x-dimension FIR filter window borders
        ifxsc = 0
        ifxec = 0

        # incremental cursors that track x-dimension IIR filter window borders
        iixsc = 0
        iixec = 0

        # incremental FIR value/mask
        inc_fir_px_val = 0
        inc_fir_px_mask = 0

        for x in range(0, width):
            #print('loop x')
            xs = max(0, x - n_size)
            xe = min(width, x + n_size)

            # Update x-dimension FIR window
            while ifxsc < xs:
                #print('dec x')
                inc_fir_px_val -= inc_fir_val[ifxsc]
                inc_fir_px_mask -= inc_fir_mask[ifxsc]
                ifxsc += 1
            while ifxec < xe:
                #print('inc x')
                inc_fir_px_val += inc_fir_val[ifxec]
                inc_fir_px_mask += inc_fir_mask[ifxec]
                ifxec += 1

            #print('val', inc_fir_px_val)
            if not mask[y, x]:
                val_count = 0
                val = 0

                val = inc_fir_px_val
                val_count = inc_fir_px_mask

                # IIR box blur over output image
                for yy in range(ys, y):
                    for xx in range(xs, x):
                        val += new_img[yy, xx]
                        val_count += 1

                if val_count > 0:
                    new_img[y, x] = val / val_count
                else:
                    new_img[y, x] = 0
            #else:
            #    new_img[y, x] = img[y, x]

    return new_img


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.warn.undeclared(True)
def optimise_rgb2(np.ndarray[UINT8DTYPE_t, ndim=2] mask,
        np.ndarray[UINT8DTYPE_t, ndim=3] img, int width, int height, int n_size):
    cdef np.ndarray[UINT8DTYPE_t, ndim=3] new_img
    cdef int x, y
    cdef int val_count, ys, ye, xs, xe, xx, yy
    cdef int r, g, b
    cdef int ifysc, ifyec, iiysc, iiyec, ifxsc, ifxec, iixsc, iixec
    cdef int inc_fir_px_r, inc_fir_px_g, inc_fir_px_b, inc_fir_px_mask

    # TODO: weights for distance?
    # This function computes a FIR and IIR version of the box blur filter incrementally
    # As seen above
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_fir_r = np.zeros([width], dtype=INTDTYPE)
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_fir_g = np.zeros([width], dtype=INTDTYPE)
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_fir_b = np.zeros([width], dtype=INTDTYPE)
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_fir_mask = np.zeros([width], dtype=INTDTYPE)

    # incremental cursors that track y-dimension FIR filter window borders
    ifysc = 0
    ifyec = 0

    # incremental cursors that track y-dimension IIR filter window borders
    iiysc = 0
    iiyec = 0

    # FIXME
    #new_img = np.empty(img.shape)
    new_img = np.copy(img)
    for y in range(0, height):
        #print('loop y')
        ys = max(0, y - n_size)
        ye = min(height, y + n_size)
        # Update y-dimension FIR window
        while ifysc < ys:
            #print('dec y')
            for x in range(0, width):
                if mask[ifysc, x]:
                    inc_fir_b[x] -= img[ifysc, x, 0]
                    inc_fir_g[x] -= img[ifysc, x, 1]
                    inc_fir_r[x] -= img[ifysc, x, 2]
                    inc_fir_mask[x] -= 1
            ifysc += 1
        while ifyec < ye:
            #print('inc y')
            for x in range(0, width):
                if mask[ifyec, x]:
                    inc_fir_b[x] += img[ifyec, x, 0]
                    inc_fir_g[x] += img[ifyec, x, 1]
                    inc_fir_r[x] += img[ifyec, x, 2]
                    inc_fir_mask[x] += 1
            ifyec += 1

        # incremental cursors that track x-dimension FIR filter window borders
        ifxsc = 0
        ifxec = 0

        # incremental cursors that track x-dimension IIR filter window borders
        iixsc = 0
        iixec = 0

        # incremental FIR value/mask
        inc_fir_px_r = 0
        inc_fir_px_g = 0
        inc_fir_px_b = 0
        inc_fir_px_mask = 0

        for x in range(0, width):
            #print('loop x')
            xs = max(0, x - n_size)
            xe = min(width, x + n_size)

            # Update x-dimension FIR window
            while ifxsc < xs:
                #print('dec x')
                inc_fir_px_b -= inc_fir_b[ifxsc]
                inc_fir_px_g -= inc_fir_g[ifxsc]
                inc_fir_px_r -= inc_fir_r[ifxsc]
                inc_fir_px_mask -= inc_fir_mask[ifxsc]
                ifxsc += 1
            while ifxec < xe:
                #print('inc x')
                inc_fir_px_b += inc_fir_b[ifxec]
                inc_fir_px_g += inc_fir_g[ifxec]
                inc_fir_px_r += inc_fir_r[ifxec]
                inc_fir_px_mask += inc_fir_mask[ifxec]
                ifxec += 1

            #print('val', inc_fir_px_val)
            if not mask[y, x]:
                val_count = 0
                r = 0

                r = inc_fir_px_r
                g = inc_fir_px_g
                b = inc_fir_px_b
                val_count = inc_fir_px_mask

                # IIR box blur over output image
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
            #else:
            #    new_img[y, x] = img[y, x]

    return new_img
