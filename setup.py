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

setup(name='internetarchivepdf',
      version=main_ns['__version__'],
      description='Internet Archive PDF compression tools',
      author='Merlijn Boris Wolf Wajer',
      author_email='merlijn@archive.org',
      packages=['internetarchivepdf'],
      scripts=['bin/recode_pdf'],
      include_package_data=True,
      package_data={'internetarchivepdf': ['data/*']},
      include_dirs=[numpy.get_include()],
      ext_modules=cythonize(['cython/sauvola.pyx', 'cython/optimiser.pyx'],
        compiler_directives={'language_level' : '3'},
      ),
      zip_safe=False,
      )
