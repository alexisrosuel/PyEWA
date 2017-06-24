from __future__ import absolute_import
from __future__ import print_function

"""
setup file
"""

import sys
from os.path import join
from setuptools import setup, Extension
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError)

NAME = 'PyEWA'
VERSION = '0.0.1'
AUTHOR = 'Alexis Rosuel'
EMAIL = 'rosuelalexis1@gmail.com'
URL = 'https://github.com/alexisrosuel/PyEWA'
DESC = 'Exponentially Weighted Aggregation in Python'
LDESC = 'PyEWA is a statistical toolkit for Python that supports Exponentially ' \
        'weighted aggregation.'
PACKAGES = ['pyewa']
PCKG_DAT = {'pyewa': ['README.md']}
REQ = ['numpy', 'scipy', 'matplotlib']


CLSF = ['Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: GIS']


setup(name=NAME, version=VERSION, author=AUTHOR, author_email=EMAIL, url=URL, description=DESC,
      long_description=LDESC, packages=PACKAGES, package_data=PCKG_DAT, classifiers=CLSF)
