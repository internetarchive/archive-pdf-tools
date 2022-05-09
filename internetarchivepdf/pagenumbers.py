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
    if v and ' ' in v:
        v = v.strip().split(' ')[0]
    if vtype == INVALID:
        return None
    elif vtype == ARABIC:
        return int(v, 10)
    elif vtype == ROMAN_LOWER or vtype == ROMAN_UPPER:
        try:
            return fromRoman(v.upper())
        except InvalidRomanNumeralError:
            raise ValueError
    elif vtype == ALPHA_LOWER or vtype == ALPHA_UPPER:
        return alpha_to_number(v.upper())
    pass


def find_next_nonzero(series):
    for v in series:
        if v is not None:
            return v
    return None


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

        if val_type in (ROMAN_UPPER, ROMAN_LOWER):
            next_val = find_next_nonzero(series[idx+1:])
            next_val_type = get_val_type(next_val)

            if val_type != next_val_type:
                if val_type == ROMAN_UPPER and next_val_type == ALPHA_UPPER or \
                        val_type == ROMAN_LOWER and next_val_type == ALPHA_LOWER:
                    val_type = next_val_type
                    val_value = get_val_value(val, val_type)

        if val_type in (ALPHA_UPPER, ALPHA_LOWER):
            next_val = find_next_nonzero(series[idx+1:])
            next_val_type = get_val_type(next_val)

            ord_val = None
            ord_next_val = None
            try:
                ord_val = ord(val)
                ord_next_val = ord(next_val)
            except TypeError:
                ord_val = None
                ord_next_val = None

            if next_val is None:
                pass
            elif ord_val is not None and ord_next_val is not None and \
                    ord_val == ord_next_val - 1:
                pass
            elif val_type == next_val_type:
                pass
            elif (val_type == ALPHA_UPPER and next_val_type == ROMAN_UPPER) or \
                 (val_type == ALPHA_LOWER and next_val_type == ROMAN_LOWER):
                     try:
                         val_type = next_val_type
                         val_value = get_val_value(val, val_type)
                     except ValueError:
                         val_type = INVALID
                         val_value = None
            elif val_type in (ALPHA_LOWER, ALPHA_UPPER) and next_val_type not in (ROMAN_UPPER, ROMAN_LOWER):
                # We can have a case where an invalid roman numeral ('XXXVIIII')
                # is followed by a arabic numeral (39), which will hit this
                # case, let's just treat it invalid
                val_type = INVALID
                val_value = None
            else:
                # This code should be unreachable now
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

        if new and idx != 0:
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
    all_series = [
        [None, 'i', 'ii', 'iii', None, None, None, 'iv', 'v', 'v', 'vi', '3',
            '4', '5', '4', '6', 'i', '7', None],
        ['i', 'ii', 'iii', 'vi', '3', '5', '4', '6', 'i', '7', 'A-2', 'B-2'],
        ['', 'i', 'ii',  'vi', '3', '5', '4', '6', 'i', '7', 'A-2', 'B-2'],
        ['i', 'j', 'k', 'l', None, None, None, None, None, None, 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX', 'XXI', None, '565', '566', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '340', '341', '342', '343', '344', '345', '346', '347', '348', '349', '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', '360', '361', '362', '363', '364', '365', '366', '367', '368', '369', '370', '371', '372', '373', '374', '375', '376', '377', '378', '379', '380', '381', '382', '383', '384', '385', '386', '387', '388', '389', '390', '391', '392', '393', '394', '395', '396', '397', '398', '399', '400', '401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416', '417', '418', '419', '420', '421', '422', '423', '424', '425', '426', '427', '428', '429', '430', '431', '432', '433', '434', '435', '436', '437', '438', '439', '440', '441', '442', '443', '444', '445', '446', '447', '448', '449', '450', '451', '452', '453', '454', '455', '456', '457', '458', '459', '460', '461', '462', '463', '464', '465', '466', '467', '468', '469', '470', '471', '472', '473', '474', '475', '476', '477', '478', '479', '480', '481', '482', '483', '484', '485', '486', '487', '488', '489', '490', '491', '492', '493', '494', '495', '496', '497', '498', '499', '500', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511', '512', '513', '514', '515', '516', '517', '518', '519', '520', '521', '522', '523', '524', '525', '526', '527', '528', '529', '530', '531', '532', '533', '534', '535', '536', '537', '538', '539', '540', '541', '542', '543', '544', '545', '546', '547', '548', '549', '550', '551', '552', '553', '554', '555', '556', '557', '558', '559', '560', '561', '562', '563', '564', 'I', None, 'II', None, 'III', None, 'IV', None, 'V', None, 'VI', None, 'VII', None, 'VII A.', None, 'VIII', None, 'IX', None, 'X', None, 'XI', None, 'XII', None, 'XIII', None, 'XIV', None, 'XV', None, 'XVI', None, 'XVII', None, 'XVIII', None, 'XIX', None, 'XX', None, 'XXI', None, 'XXII', None, 'XXIII', None, 'XXIV', None, 'XXV', None, 'XXVI', None, 'XXVII', None, 'XXVIII', None, 'XXIX', None, 'XXX', None, 'XXXI', None, 'XXXII', None, 'XXXIII', None, 'XXXIV', None, None, None, None, None, None, None]
    ]

    for series in all_series:
        res, all_ok = parse_series(series)
        from pprint import pprint
        pprint(res)
        print(all_ok)
        #print(series_to_pdf(res))
