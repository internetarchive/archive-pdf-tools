from setuptools import setup
from distutils.util import convert_path

from Cython.Build import cythonize
import numpy

# Work around bugs in passing CFLAGS to the Cython compilation process
import os
os.environ['CFLAGS'] = '-Ofast -DNPY_NO_DEPRECATED_API'

setup(
    include_dirs = [numpy.get_include()],
    ext_modules = cythonize(
        ['cython/sauvola.pyx', 'cython/optimiser.pyx'],
        compiler_directives = {'language_level' : '3'},
    ),
)
