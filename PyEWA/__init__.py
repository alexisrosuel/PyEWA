__author__ = 'Alexis Rosuel'
__version__ = '0.0.1'
__doc__ = """Code by Alexis Rosuel
rosuelalexis1@gmail.com
Dependencies:
    numpy
    scipy
    matplotlib

Modules:
    online_learning: contains the main script
    test: Contains the test script.
References:
    S. Gerchinovitz, Prediction of individual sequences and prediction in the statistical framework:
    some links around sparse regression and aggregation techniques, PhD Thesis, Univ. Paris 11, 2011. (Chapters 2 and 3).

Copyright (c) 2017-2017 Alexis Rosuel
"""

from . import *

from .examples import *
from .tests import test

__all__ = ['EWA']
