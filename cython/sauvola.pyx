# Code inspired / ported from:
#
# 1. https://arxiv.org/pdf/1905.13038.pdf
# 2. https://github.com/chungkwong/binarizer
#
# The Sauvola specific formula is repeated four times in the code because I
# didn't get function calls to work fast in Cython (40s for a 15000x15000 images instead of 4s)
#
# Author: Merlijn Wajer <merlijn@archive.org>
# License: AGPL-v3

import numpy as np
cimport numpy as np
cimport cython

INTDTYPE = np.int
UINT8DTYPE = np.uint8

ctypedef np.int_t INTDTYPE_t
ctypedef np.uint8_t UINT8DTYPE_t

# Speed up the code
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
# Let's make sure all variables have a c type, otherwise performs goes out of
# the window
@cython.warn.undeclared(True)
def binarise_sauvola(np.ndarray[UINT8DTYPE_t, ndim=1] in_arr, np.ndarray[UINT8DTYPE_t, ndim=1] out_arr, int width, int height, int window_width, int window_height, double k, double R):
    """
    Perform fast Sauvola binarisation on the given input array/image.

    Args:

    * in_arr (numpy.ndarray[numpy.uint8, ndim=1]): input array, flattened
    * our_arr (numpy.ndarray[numpy.uint8, ndim=1]): output array, flattened, must be allocated already
    * width (int): width of mask
    * height int): height of mask
    * window_width (int): Sauvola window width
    * window_height(int): Sauvola window height
    * k (double): k parameter
    * R (double): R parameter

    Example usage:

    >>> h, w = img.shape
    >>> out_img = np.ndarray(img.shape, dtype=np.bool)
    >>> out_img = np.reshape(out_img, w*h)
    >>> in_img = np.reshape(img, w*h)
    >>> binarise_sauvola(in_img, out_img, w, h, window_size, window_size, k, R)
    >>> out_img = np.reshape(out_img, (h, w))
    >>> out_img = np.invert(out_img)
    """
    cdef int i, l, r, o, u, j, dr0, dr1, dr2, dr3
    cdef int index_top, index_bottom, win_top, win_bottom, index, bottom_max
    cdef int top, bottom, win_height, sum_, count, win_right, win_left, imax
    cdef long square_sum

    cdef char formres;
    cdef double mean, variance, tmp

    cdef double k2=k*k/R/R;

    cdef np.ndarray[INTDTYPE_t, ndim=1] integral = np.zeros([width], dtype=INTDTYPE)
    cdef np.ndarray[INTDTYPE_t, ndim=1] integral_square = np.zeros([width], dtype=INTDTYPE)

    cdef INTDTYPE_t pixel


    for i in range(0, width):
        integral[i] = 0
        integral_square[i] = 0

    l = (window_width + 1) / 2
    r = window_width / 2;
    o = (window_height + 1) / 2
    u = window_height / 2

    index = 0
    imax = min(height, u)
    for i in range(0, imax):
        for j in range(0, width):
            pixel = in_arr[index]

            integral[j] += pixel
            integral_square[j] += pixel * pixel

            index += 1

    dr0 = min(window_width, width)
    dr1 = min(r, width)
    dr2 = width - l
    dr3 = min(dr2, 0)
    index_top = 0
    index_bottom = u * width

    win_top = -o
    win_bottom = u
    index = 0
    bottom_max = height + u
    while win_bottom < bottom_max:
        if win_top>=0:
            for j in range(0, width):
                pixel = in_arr[index_top]

                integral[j] -= pixel
                integral_square[j] -= pixel * pixel

                index_top += 1

            top = win_top
        else:
            top = -1

        if win_bottom < height:
            for j in range(0, width):
                pixel = in_arr[index_bottom]

                integral[j] += pixel
                integral_square[j] += pixel*pixel

                index_bottom += 1

            bottom = win_bottom
        else:
            bottom = height - 1;

        win_height = bottom-top
        sum_ = 0
        square_sum = 0
        for j in range(0, dr1):
            sum_ += integral[j];
            square_sum += integral_square[j];

        count = dr1 * win_height;
        win_right = r;
        while win_right < dr0:
            count += win_height;
            sum_ += integral[win_right];
            square_sum += integral_square[win_right];

            pixel = in_arr[index]
            if k >= 0:
                mean = sum_ / count
                variance = square_sum / count - (mean * mean)
                tmp = (pixel + (mean * (k- 1)))
                formres = ((tmp <= 0) or (tmp * tmp <= mean * mean * k2 * variance))
            else:
                mean = sum_ / count
                variance = square_sum / count - (mean*mean)
                tmp = (pixel + mean * (k - 1))
                formres = (tmp <= 0 and tmp * tmp >= mean * mean * k2 * variance)
            out_arr[index] = 0 if formres else 1

            win_right += 1
            index += 1

        win_left = win_right - window_width;
        if win_right >= width:
            while win_left < dr3:
                pixel = in_arr[index]
                if k >= 0:
                    mean = sum_ / count
                    variance = square_sum / count - (mean * mean)
                    tmp = (pixel + (mean * (k- 1)))
                    formres = ((tmp <= 0) or (tmp * tmp <= mean * mean * k2 * variance))
                else:
                    mean = sum_ / count
                    variance = square_sum / count - (mean*mean)
                    tmp = (pixel + mean * (k - 1))
                    formres = (tmp <= 0 and tmp * tmp >= mean * mean * k2 * variance)
                out_arr[index] = 0 if formres else 1

                win_left += 1
                index += 1
        else:
            while win_right < width:
                sum_ += integral[win_right] - integral[win_left];
                square_sum += integral_square[win_right] - integral_square[win_left];

                pixel = in_arr[index]
                if k >= 0:
                    mean = sum_ / count
                    variance = square_sum / count - (mean * mean)
                    tmp = (pixel + (mean * (k- 1)))
                    formres = ((tmp <= 0) or (tmp * tmp <= mean * mean * k2 * variance))
                else:
                    mean = sum_ / count
                    variance = square_sum / count - (mean*mean)
                    tmp = (pixel + mean * (k - 1))
                    formres = (tmp <= 0 and tmp * tmp >= mean * mean * k2 * variance)
                out_arr[index] = 0 if formres else 1

                win_left += 1
                win_right += 1
                index += 1

        while win_left < dr2:
            count -= win_height;
            sum_ -= integral[win_left];
            square_sum -= integral_square[win_left];

            pixel = in_arr[index]
            if k >= 0:
                mean = sum_ / count
                variance = square_sum / count - (mean * mean)
                tmp = (pixel + (mean * (k- 1)))
                formres = ((tmp <= 0) or (tmp * tmp <= mean * mean * k2 * variance))
            else:
                mean = sum_ / count
                variance = square_sum / count - (mean*mean)
                tmp = (pixel + mean * (k - 1))
                formres = (tmp <= 0 and tmp * tmp >= mean * mean * k2 * variance)
            out_arr[index] = 0 if formres else 1

            win_left += 1
            index += 1

        win_top += 1
        win_bottom += 1

    return 0
