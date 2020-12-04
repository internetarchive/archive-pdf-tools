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


def scandata_xml_get_page_numbers(xml_file):
    scandata = xmltodict.parse(open(xml_file, 'rb'))

    res = []

    pages = scandata['book']['pageData']['page']

    # If there is just one page, pages is not a list.
    if not isinstance(pages, list):
        pages = [pages]
    for idx in range(len(pages)):
        try:
            add_to_access_format = pages[idx]['addToAccessFormats']
            if add_to_access_format == 'false':
                continue
        except KeyError:
            pass

        pno = pages[idx].get('pageNumber', None)
        res.append(pno)

    return res


def scandata_xml_get_dpi_per_page(xml_file):
    scandata = xmltodict.parse(open(xml_file, 'rb'))

    res = []

    pages = scandata['book']['pageData']['page']

    # If there is just one page, pages is not a list.
    if not isinstance(pages, list):
        pages = [pages]
    for idx in range(len(pages)):
        try:
            add_to_access_format = pages[idx]['addToAccessFormats']
            if add_to_access_format == 'false':
                continue
        except KeyError:
            pass

        ppi = pages[idx].get('ppi', None)
        res.append(ppi)

    return res


def scandata_xml_get_document_dpi(xml_file):
    scandata = xmltodict.parse(open(xml_file, 'rb'))

    doc_ppi = scandata['book']['bookData'].get('dpi', None)

    if doc_ppi is not None:
        try:
            doc_ppi = int(doc_ppi)
        except (ValueError):
            return None

    return doc_ppi
