import re


def normalize_pdf_for_comparison(pdf_bytes):
    """Strip metadata that varies between runs (dates, IDs) for byte-exact comparison."""
    # Replace creation/modification dates
    # Date format: D:YYYYMMDDHHmmSS or D:YYYYMMDDHHmmSS+HH'mm'
    result = re.sub(rb"/CreationDate \(D:\d{14}(?:\+\d{2}\x27\d{2})?\)", b'/CreationDate (D:20000101000000Z)', pdf_bytes)
    result = re.sub(rb"/ModDate \(D:\d{14}(?:\+\d{2}\x27\d{2})?\)", b'/ModDate (D:20000101000000Z)', result)
    # Replace /ID in trailer
    result = re.sub(rb'/ID \[<[0-9a-fA-F]+> <[0-9a-fA-F]+>\]', b'/ID [<00000000000000000000000000000000> <00000000000000000000000000000000>]', result)
    return result
