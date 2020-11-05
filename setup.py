from setuptools import setup

setup(name='internetarchivepdf',
      version='0.0.1',
      description='Internet Archive PDF compression tools',
      author='Merlijn Boris Wolf Wajer',
      author_email='merlijn@archive.org',
      packages=['internetarchivepdf'],
      scripts=['bin/recode_pdf'],
      include_package_data=True,
      package_data={'internetarchivepdf': ['data/*']})
