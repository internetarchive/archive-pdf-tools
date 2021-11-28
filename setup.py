from setuptools import setup
from distutils.util import convert_path

from Cython.Build import cythonize
import numpy

# Work around bugs in passing CFLAGS to the Cython compilation process
import os
os.environ['CFLAGS'] = '-Ofast -DNPY_NO_DEPRECATED_API'

main_ns = {}
ver_path = convert_path('internetarchivepdf/const.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

version = main_ns['__version__']
setup(name='archive-pdf-tools',
      version=version,
      packages=['internetarchivepdf'],
      description='Internet Archive PDF compression tools',
      author='Merlijn Boris Wolf Wajer',
      author_email='merlijn@archive.org',
      url='https://github.com/internetarchive/archive-pdf-tools',
      download_url='https://github.com/internetarchive/archive-pdf-tools/archive/%s.tar.gz' % version,
        keywords=['PDF', 'MRC', 'hOCR', 'Internet Archive'],
      license='AGPL-3.0',
      scripts=['bin/recode_pdf', 'tools/mrcview', 'tools/maskview', 'tools/pdfimagesmrc'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: GNU Affero General Public License v3',
          'Programming Language :: Python :: 3',
      ],
      python_requires='>=3.6',
      include_package_data=True,
      package_data={'internetarchivepdf': ['data/*']},
      include_dirs=[numpy.get_include()],
      install_requires=[
            'PyMuPDF==1.19.2',
            'numpy',
            'lxml',
            'scikit-image',
            'Pillow==8.3.2',
            'roman==3.3',
            'xmltodict==0.12.0',
            'archive-hocr-tools==1.1.13',
      ],
      ext_modules=cythonize(['cython/sauvola.pyx', 'cython/optimiser.pyx'],
        compiler_directives={'language_level' : '3'},
      ),
      zip_safe=False,
      )
