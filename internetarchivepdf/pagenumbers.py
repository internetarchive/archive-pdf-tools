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

import re

# TODO:
# - Implement page number prefix
# - Deal with ambiguity between roman numerals 'I', 'II', 'MM' and Alpha 'I',
#   'II', etc.

from roman import fromRoman, InvalidRomanNumeralError

INVALID, ARABIC, ROMAN_LOWER, ROMAN_UPPER, ALPHA_UPPER, ALPHA_LOWER = range(6)

_type2str = {
        INVALID: 'Invalid',
        ARABIC: 'Arabic',
        ROMAN_LOWER: 'Roman lower',
        ROMAN_UPPER: 'Roman upper',
        ALPHA_UPPER: 'Alpha upper',
        ALPHA_LOWER: 'Alpha lower'
}

_type2pdf = {
        ARABIC: '/D',
        ROMAN_LOWER: '/r',
        ROMAN_UPPER: '/R',
        ALPHA_UPPER: '/A',
        ALPHA_LOWER: '/a',
}

_ARABIC_RE = re.compile('[0-9]+')
_ALPHA_UPPER_RE = re.compile('[A-Z]+')
_ALPHA_LOWER_RE = re.compile('[a-z]+')


def is_arabic(n):
    return _ARABIC_RE.match(n) and n.isnumeric()


def is_alpha_upper(n):
    return _ALPHA_UPPER_RE.match(n)


def is_alpha_lower(n):
    return _ALPHA_LOWER_RE.match(n)


def is_roman(n):
    try:
        _ = fromRoman(n.upper())
        return True
    except InvalidRomanNumeralError:
        return False


def is_roman_lower(n):
    return n.lower() == n and is_roman(n)


def is_roman_upper(n):
    return n.upper() == n and is_roman(n)


# Evince goes from 'Z' to 'AA' to 'ZZ' to 'AAA'.
def alpha_to_number(n):
    # 'A' = 1
    # 'Z' = 26
    # 'AA' = 27
    # 'ZZ' = 52
    # 'AAA' = 53

    first = True
    res = 1
    for c in n:
        tmp = ord(c) - ord('A')
        res += tmp

        if first:
            first = False
        else:
            res += 26 - tmp

    return res


def get_val_type(v):
    if v is None:
        # XXX: Does this special case make sense?
        return INVALID
    elif is_arabic(v):
        return ARABIC
    # Prefer roman over alpha since we do not yet support alpha
    elif is_roman_lower(v):
        return ROMAN_LOWER
    elif is_roman_upper(v):
        return ROMAN_UPPER
    elif is_alpha_upper(v):
        return ALPHA_UPPER
    elif is_alpha_lower(v):
        return ALPHA_LOWER
    else:
        raise ValueError('Page number not in spec: %s' % repr(v))


def get_val_value(v, vtype):
    if vtype == INVALID:
        return None
    elif vtype == ARABIC:
        return int(v, 10)
    elif vtype == ROMAN_LOWER or vtype == ROMAN_UPPER:
        return fromRoman(v.upper())
    elif vtype == ALPHA_LOWER or vtype == ALPHA_UPPER:
        return alpha_to_number(v.upper())
    pass


def parse_series(series):
    # Let's first split up the series:
    # - detect when the numbers are not in sequence
    # - detect when the type changes
    last_value = None
    last_val_type = INVALID

    series_start = 0

    resulting_series = []
    all_ok = True

    running_series = []
    running_series_n = []

    for idx, val in enumerate(series):
        new = False

        try:
            val_type = get_val_type(val)
            val_value = get_val_value(val, val_type)
        except ValueError as e:
            all_ok = False
            val_type = INVALID
            val_value = None

        if val_type in(ALPHA_UPPER, ALPHA_LOWER):
            raise ValueError('Alpha page numbers are not supported at the '
                             ' moment due to ambiguity in the spec.')

        if val_type != last_val_type:
            new = True

        if val_type == INVALID and last_val_type == INVALID:
            pass
        else:
            if last_val_type == INVALID:
                new = True
            elif val_type == INVALID:
                new = True
            elif val_value != last_value + 1:
                new = True

        if new:
            resulting_series.append({'start': series_start,
                                     'type': last_val_type,
                                     'type_human': _type2str[last_val_type],
                                     'values': running_series,
                                     'values_numeric': running_series_n})

            series_start = idx
            running_series = []
            running_series_n = []

        running_series.append(val)
        running_series_n.append(val_value)

        last_value = val_value
        last_val_type = val_type

    resulting_series.append({'start': series_start,
                             'type': last_val_type,
                             'type_human': _type2str[last_val_type],
                             'values': running_series,
                             'values_numeric': running_series_n})

    return resulting_series, all_ok


# https://www.w3.org/TR/WCAG20-TECHS/PDF17.html
# Page labels are specified as follows:
# 
#     /S specifies the numbering style for page numbers:
# 
#         /D - Arabic numerals (1,2,3...)
# 
#         /r - lowercase Roman numerals (i, ii, iii,...)
# 
#         /R - uppercase Roman numerals (I, II, III,...)
# 
#         /A - uppercase letters (A-Z)
# 
#         /a - lowercase letters (a-z)
# 
#     /P (optional) - page number prefix
# 
#     /St (optional) - the value of the first page number in the range (default:
#     1)
def series_to_pdf(series):
    res = '''  /PageLabels <<
    /Nums [ '''

    for s in series:
        r = '%d ' % s['start']
        if s['type'] == INVALID:
            r += '<<\n        >> '
        else:
            r += '<<\n'
            r += '         /S ' + _type2pdf[s['type']] + '\n'
            r += '         /St %d' % s['values_numeric'][0] + '\n'
            r += '        '
            r += '>> '

        res += r

    res += ''']
    >>'''

    return res


if __name__ == '__main__':
    series = [None, 'i', 'ii', 'iii', None, None, None, 'iv', 'v', 'v', 'vi',
            '3', '4', '5', '4', '6', 'i', '7', None]
    #series = ['', 'i', 'ii', 'vi', 3, 5, 4, 6, 'i', 7, 'A-2', 'B-2']

    res, all_ok = parse_series(series)
    print(res)
    print(all_ok)
    print(series_to_pdf(res))
