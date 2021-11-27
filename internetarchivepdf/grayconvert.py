# archive-pdf-tools
# Copyright (C) 2020-2021, Internet Archive
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Author: Merlijn Boris Wolf Wajer <merlijn@archive.org>

import numpy as np
from skimage.color import rgb2hsv

perc2val = lambda x: (x*255)/100

def level_arr(arr, minv=0, maxv=255):
    interval = (maxv/255.) - (minv/255.)
    arr_zero = arr < minv
    arr_max = arr > maxv
    arr[::] = ((arr[::] - minv) / interval)
    arr[arr_zero] = 0
    arr[arr_max] = 255
    return arr


# Straight forward port of color2Gray.sh script
# We might be able to do better, but there are only a few users of this script
# in the archive.org currently, so more time has not been invested in finding
# alternative or better ways.
def special_gray_convert(imd):
    components = ('r', 'g', 'b')

    d = {}
    for i, k in enumerate(components):
        for fun in ['min', 'max', 'mean', 'std']:
            d[k + '_' + fun] = getattr(np, fun)(imd[:,:,i]) / 255.

    bright_adjust = round(d['r_mean'] * d['g_mean'] * d['b_mean'] /
                    (d['b_max']*(1-d['r_std'])*(1-d['g_std'])*(1-d['b_std'])), 4)

    low_thres = min(int((196 * d['r_min']+14.5)/1), 50)

    high_thres = {
            'r': min(int((35.66*bright_adjust+48.5)/1), 95),
            'g': min(int((39.22*bright_adjust+44.5)/1), 95),
            'b': min(int((45.16*bright_adjust+36.5)/1), 95),
            }

    new_imd = np.copy(imd)
    for i, c in enumerate(components):
        new_imd[:,:,i] = level_arr(new_imd[:,:,i],
                                   minv=perc2val(low_thres),
                                   maxv=perc2val(high_thres[c]))

    hsv = rgb2hsv(new_imd)
    # Calculate the 'L' from 'HSL' as L = S * (1 - V/2)
    l = hsv[:,:,2] * (1 - (hsv[:,:,1]/2))
    return np.array(l * 255, dtype=np.uint8)


