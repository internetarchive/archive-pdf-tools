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
    """
    "Optimises" an input image for JPEG2000 compression, it does this by
    radiating pixels in the mask to pixels not in the mask, downwards and to the
    right. This (hopefully) allows for more optimal lossy compression of the
    pixels in the mask.

    Grayscale version.

    Args:

    * mask (numpy.ndarray[numpy.uint8, ndim=2]): input mask
    * img: (numpy.ndarray[numpy.uint8, ndim=2): input image
    * width (int): mask/img width
    * height (int): mask/img height
    * n_size: window size

    Returns a new image (numpy.ndarray[numpy.uint8, ndim=2])
    """
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
    """
    "Optimises" an input image for JPEG2000 compression, it does this by
    radiating pixels in the mask to pixels not in the mask, downwards and to the
    right. This (hopefully) allows for more optimal lossy compression of the
    pixels in the mask.

    RGB version.

    Args:

    * mask (numpy.ndarray[numpy.uint8, ndim=2]): input mask
    * img: (numpy.ndarray[numpy.uint8, ndim=3): input image
    * width (int): mask/img width
    * height (int): mask/img height
    * n_size: window size

    Returns a new image (numpy.ndarray[numpy.uint8, ndim=3])
    """
    cdef np.ndarray[UINT8DTYPE_t, ndim=3] new_img
    cdef int x, y
    cdef int val_count, ys, ye, xs, xe, xx, yy
    cdef int r, g, b

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
                            r += img[yy, xx, 0]
                            g += img[yy, xx, 1]
                            b += img[yy, xx, 2]
                            val_count += 1

                for yy in range(ys, y):
                    for xx in range(xs, x):
                        r += new_img[yy, xx, 0]
                        g += new_img[yy, xx, 1]
                        b += new_img[yy, xx, 2]
                        val_count += 1

                if val_count > 0:
                    new_img[y, x, 0] = r / val_count
                    new_img[y, x, 1] = g / val_count
                    new_img[y, x, 2] = b / val_count
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
    """
    "Optimises" an input image for JPEG2000 compression, it does this by
    radiating pixels in the mask to pixels not in the mask, downwards and to the
    right. This (hopefully) allows for more optimal lossy compression of the
    pixels in the mask.

    Fast and grayscale version.

    Args:

    * mask (numpy.ndarray[numpy.uint8, ndim=2]): input mask
    * img: (numpy.ndarray[numpy.uint8, ndim=2): input image
    * width (int): mask/img width
    * height (int): mask/img height
    * n_size: window size

    Returns a new image (numpy.ndarray[numpy.uint8, ndim=2])
    """
    cdef np.ndarray[UINT8DTYPE_t, ndim=2] new_img
    cdef int x, y
    cdef int val_count, val, ys, ye, xs, xe, xx, yy
    cdef int ifysc, ifyec, iiysc, iiyec, ifxsc, ifxec, iixsc, iixec
    cdef int inc_fir_px_val, inc_fir_px_mask
    cdef int inc_iir_px_val
    cdef int iir_window_size

    # This function computes a FIR and IIR version of the box blur filter incrementally
    # As seen above
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_fir_val = np.zeros([width], dtype=INTDTYPE)
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_fir_mask = np.zeros([width], dtype=INTDTYPE)
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_iir_val = np.zeros([width], dtype=INTDTYPE)

    # incremental cursors that track y-dimension FIR filter window borders
    ifysc = 0
    ifyec = 0

    # incremental cursors that track y-dimension IIR filter window borders
    iiysc = 0
    iiyec = 0

    new_img = np.copy(img)
    for y in range(0, height):
        #print('loop y')
        ys = max(0, y - n_size)
        ye = min(height, y + n_size)
        # Update y-dimension FIR window
        while ifysc < ys:
            for x in range(0, width):
                if mask[ifysc, x]:
                    inc_fir_val[x] -= img[ifysc, x]
                    inc_fir_mask[x] -= 1
            ifysc += 1
        while ifyec < ye:
            for x in range(0, width):
                if mask[ifyec, x]:
                    inc_fir_val[x] += img[ifyec, x]
                    inc_fir_mask[x] += 1
            ifyec += 1
        while iiysc < ys:
            for x in range(0, width):
                inc_iir_val[x] -= new_img[iiysc, x]
            iiysc += 1
        while iiyec < y:
            for x in range(0, width):
                inc_iir_val[x] += new_img[iiyec, x]
            iiyec += 1

        # incremental cursors that track x-dimension FIR filter window borders
        ifxsc = 0
        ifxec = 0

        # incremental cursors that track x-dimension IIR filter window borders
        iixsc = 0
        iixec = 0

        # incremental FIR value/mask
        inc_fir_px_val = 0
        inc_fir_px_mask = 0

        # incremental IIR value
        inc_iir_px_val = 0

        for x in range(0, width):
            xs = max(0, x - n_size)
            xe = min(width, x + n_size)

            # Update x-dimension FIR window
            while ifxsc < xs:
                inc_fir_px_val -= inc_fir_val[ifxsc]
                inc_fir_px_mask -= inc_fir_mask[ifxsc]
                ifxsc += 1
            while ifxec < xe:
                inc_fir_px_val += inc_fir_val[ifxec]
                inc_fir_px_mask += inc_fir_mask[ifxec]
                ifxec += 1
            while iixsc < xs:
                inc_iir_px_val -= inc_iir_val[iixsc]
                iixsc += 1
            while iixec < x:
                inc_iir_px_val += inc_iir_val[iixec]
                iixec += 1

            if not mask[y, x]:
                val_count = 0
                val = 0

                iir_window_size = (y - ys) * (x - xs)

                val = inc_fir_px_val + inc_iir_px_val
                val_count = inc_fir_px_mask + iir_window_size

                if val_count > 0:
                    new_img[y, x] = val / val_count
                else:
                    new_img[y, x] = 0
            #else:
            #    new_img[y, x] = img[y, x]

    return new_img


#@cython.boundscheck(False)
#@cython.wraparound(False)
@cython.cdivision(True)
@cython.warn.undeclared(True)
def optimise_rgb2(np.ndarray[UINT8DTYPE_t, ndim=2] mask,
        np.ndarray[UINT8DTYPE_t, ndim=3] img, int width, int height, int n_size):
    """
    "Optimises" an input image for JPEG2000 compression, it does this by
    radiating pixels in the mask to pixels not in the mask, downwards and to the
    right. This (hopefully) allows for more optimal lossy compression of the
    pixels in the mask.

    Fast and RGB version

    Args:

    * mask (numpy.ndarray[numpy.uint8, ndim=2]): input mask
    * img: (numpy.ndarray[numpy.uint8, ndim=3): input image
    * width (int): mask/img width
    * height (int): mask/img height
    * n_size: window size

    Returns a new image (numpy.ndarray[numpy.uint8, ndim=3])
    """
    cdef np.ndarray[UINT8DTYPE_t, ndim=3] new_img
    cdef int x, y
    cdef int val_count, ys, ye, xs, xe, xx, yy
    cdef int r, g, b
    cdef int ifysc, ifyec, iiysc, iiyec, ifxsc, ifxec, iixsc, iixec
    cdef int inc_fir_px_r, inc_fir_px_g, inc_fir_px_b, inc_fir_px_mask
    cdef int inc_iir_px_r, inc_iir_px_g, inc_iir_px_b
    cdef int iir_window_size

    # This function computes a FIR and IIR version of the box blur filter incrementally
    # As seen above
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_fir_r = np.zeros([width], dtype=INTDTYPE)
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_fir_g = np.zeros([width], dtype=INTDTYPE)
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_fir_b = np.zeros([width], dtype=INTDTYPE)
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_iir_r = np.zeros([width], dtype=INTDTYPE)
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_iir_g = np.zeros([width], dtype=INTDTYPE)
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_iir_b = np.zeros([width], dtype=INTDTYPE)
    cdef np.ndarray[INTDTYPE_t, ndim=1] inc_fir_mask = np.zeros([width], dtype=INTDTYPE)

    # incremental cursors that track y-dimension FIR filter window borders
    ifysc = 0
    ifyec = 0

    # incremental cursors that track y-dimension IIR filter window borders
    iiysc = 0
    iiyec = 0

    new_img = np.copy(img)
    for y in range(0, height):
        ys = max(0, y - n_size)
        ye = min(height, y + n_size)
        # Update y-dimension FIR window
        while ifysc < ys:
            for x in range(0, width):
                if mask[ifysc, x]:
                    inc_fir_r[x] -= img[ifysc, x, 0]
                    inc_fir_g[x] -= img[ifysc, x, 1]
                    inc_fir_b[x] -= img[ifysc, x, 2]
                    inc_fir_mask[x] -= 1
            ifysc += 1
        while ifyec < ye:
            for x in range(0, width):
                if mask[ifyec, x]:
                    inc_fir_r[x] += img[ifyec, x, 0]
                    inc_fir_g[x] += img[ifyec, x, 1]
                    inc_fir_b[x] += img[ifyec, x, 2]
                    inc_fir_mask[x] += 1
            ifyec += 1
        while iiysc < ys:
            for x in range(0, width):
                inc_iir_r[x] -= new_img[iiysc, x, 0]
                inc_iir_g[x] -= new_img[iiysc, x, 1]
                inc_iir_b[x] -= new_img[iiysc, x, 2]
            iiysc += 1
        while iiyec < y:
            for x in range(0, width):
                inc_iir_r[x] += new_img[iiyec, x, 0]
                inc_iir_g[x] += new_img[iiyec, x, 1]
                inc_iir_b[x] += new_img[iiyec, x, 2]
            iiyec += 1

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

        # incremental IIR value
        inc_iir_px_r = 0
        inc_iir_px_g = 0
        inc_iir_px_b = 0

        for x in range(0, width):
            xs = max(0, x - n_size)
            xe = min(width, x + n_size)

            # Update x-dimension FIR window
            while ifxsc < xs:
                inc_fir_px_r -= inc_fir_r[ifxsc]
                inc_fir_px_g -= inc_fir_g[ifxsc]
                inc_fir_px_b -= inc_fir_b[ifxsc]
                inc_fir_px_mask -= inc_fir_mask[ifxsc]
                ifxsc += 1
            while ifxec < xe:
                inc_fir_px_r += inc_fir_r[ifxec]
                inc_fir_px_g += inc_fir_g[ifxec]
                inc_fir_px_b += inc_fir_b[ifxec]
                inc_fir_px_mask += inc_fir_mask[ifxec]
                ifxec += 1
            while iixsc < xs:
                inc_iir_px_r -= inc_iir_r[iixsc]
                inc_iir_px_g -= inc_iir_g[iixsc]
                inc_iir_px_b -= inc_iir_b[iixsc]
                iixsc += 1
            while iixec < x:
                inc_iir_px_r += inc_iir_r[iixec]
                inc_iir_px_g += inc_iir_g[iixec]
                inc_iir_px_b += inc_iir_b[iixec]
                iixec += 1

            if not mask[y, x]:
                val_count = 0

                iir_window_size = (y - ys) * (x - xs)

                b = inc_fir_px_b + inc_iir_px_b
                g = inc_fir_px_g + inc_iir_px_g
                r = inc_fir_px_r + inc_iir_px_r
                val_count = inc_fir_px_mask + iir_window_size

                if val_count > 0:
                    new_img[y, x, 0] = r / val_count
                    new_img[y, x, 1] = g / val_count
                    new_img[y, x, 2] = b / val_count
                else:
                    new_img[y, x, 0] = 0
                    new_img[y, x, 1] = 0
                    new_img[y, x, 2] = 0
            #else:
            #    new_img[y, x] = img[y, x]

    return new_img


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.warn.undeclared(True)
def fast_mask_denoise(np.ndarray[UINT8DTYPE_t, ndim=2] mask, int width, int
        height, int mincnt, int n_size):
    """
    Fast in-place denoiser, focussed on speed rather than quality.

    This function removes noise based on the amount of pixels in its
    neighbourhood

    Args:

    * mask (numpy.ndarray[numpy.uint8, ndim=2]): mask, modified in place
    * width (int): width of mask
    * height int): height of mask
    * mincnt (int): min pixels in neighbourhood to not be counted as noise
    * n_size (int): neighbourhood size in all x and y (-n_size, +n_size)

    Returns the input mask, denoised.
    """
    cdef np.ndarray[UINT8DTYPE_t, ndim=2] new_img
    cdef int x, y
    cdef int cnt = 0;
    cdef int ww = 0;
    cdef int hh = 0;

    for y in range(n_size, height - n_size):
        for x in range(n_size, width - n_size):
            if mask[y, x]:
                cnt = 0
                for hh in range(-n_size, n_size + 1):
                    for ww in range(-n_size, n_size + 1):
                        cnt += mask[y - hh, x - ww]
                # We count the current pixel with (hh=0, ww=0),
                # so subtract one from cnt,
                # this is faster than branching in the loop
                mask[y, x] = (cnt - 1) >= mincnt

    return mask
