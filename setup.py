from setuptools import setup
from distutils.util import convert_path

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
      package_data={'internetarchivepdf': ['data/*']})
