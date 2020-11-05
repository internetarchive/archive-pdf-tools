import xmltodict

def scandata_xml_get_skip_pages(xml_file):
    scandata = xmltodict.parse(open(xml_file, 'rb'))

    skip = []

    for idx in range(len(scandata['book']['pageData']['page'])):
        try:
            add_to_access_format = scandata['book']['pageData']['page'][idx]['addToAccessFormats']
            if add_to_access_format == 'false':
                skip.append(idx)
        except KeyError:
            pass

    return skip
