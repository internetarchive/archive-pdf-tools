import pytest
from os.path import join, exists
from PIL import Image


def read_file(tmp_path_factory, name):
    """Copy a test file from test-files/ to a temporary location and return its path."""
    src = join('test-files', name)
    if not exists(src):
        pytest.skip(f"Test data file not found: {src}. "
                     "You need to place the required test files in test-files/.")
    dst = tmp_path_factory.getbasetemp() / name
    with open(src, 'rb') as fp:
        dst.write_bytes(fp.read())
    return str(dst)


@pytest.fixture(scope='session')
def sim_hocr_file(tmp_path_factory):
    """Fixture for the companion archive-hocr-tools hOCR test file."""
    return read_file(tmp_path_factory, 'sim_english-illustrated-magazine_1884-12_2_15_hocr.html')


@pytest.fixture(scope='session')
def sim_english_pagenumber_json_file(tmp_path_factory):
    """Fixture for the companion archive-hocr-tools page numbers test file."""
    return read_file(tmp_path_factory, 'sim_english_pagenumbers.json')


@pytest.fixture(scope='session')
def test_img_sim_english(tmp_path_factory):
    """Microfilm test image (grayscale, 2414x3560)."""
    path = read_file(tmp_path_factory, 'test_img_sim_english.jp2')
    return Image.open(path)


@pytest.fixture(scope='session')
def test_img_microfiche(tmp_path_factory):
    """Microfiche test image (grayscale, 4096x350)."""
    path = read_file(tmp_path_factory, 'test_img_microfiche.jp2')
    return Image.open(path)


@pytest.fixture(scope='session')
def test_img_alienate(tmp_path_factory):
    """Color book test image (RGB, 4000x6000)."""
    path = read_file(tmp_path_factory, 'test_img_alienate.jp2')
    return Image.open(path)


@pytest.fixture(scope='session')
def test_img_oxford(tmp_path_factory):
    """Oxford Duden pictorial book test image (RGB, 1918x3057)."""
    path = read_file(tmp_path_factory, 'test_img_oxford.jp2')
    return Image.open(path)


@pytest.fixture(scope='session')
def test_mask_array(tmp_path_factory):
    """Fixture for a boolean mask array for testing mask encoding."""
    import numpy as np
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    return mask


@pytest.fixture(scope='session')
def test_hocr_data_sim_english(tmp_path_factory):
    """hOCR data for the sim_english microfilm item (single page matching test_img_sim_english)."""
    path = read_file(tmp_path_factory, 'test_hocr_sim_english_page1.html')
    with open(path) as f:
        return f.read()


@pytest.fixture(scope='session')
def test_hocr_data_microfiche(tmp_path_factory):
    """hOCR data for the microfiche item."""
    path = read_file(tmp_path_factory, 'test_hocr_microfiche.html')
    with open(path) as f:
        return f.read()


@pytest.fixture(scope='session')
def test_hocr_data_alienate(tmp_path_factory):
    """hOCR data for the alienate color book item."""
    path = read_file(tmp_path_factory, 'test_hocr_alienate.html')
    with open(path) as f:
        return f.read()


@pytest.fixture(scope='session')
def test_hocr_data_oxford(tmp_path_factory):
    """hOCR data for the Oxford Duden pictorial book item."""
    path = read_file(tmp_path_factory, 'test_hocr_oxford.html')
    with open(path) as f:
        return f.read()


@pytest.fixture(scope='session')
def test_hocr_data_alienate_page1(tmp_path_factory):
    """Single-page hOCR data for the alienate color book item (page 0, matching test_img_alienate)."""
    path = read_file(tmp_path_factory, 'test_hocr_alienate_page1.html')
    with open(path) as f:
        return f.read()
