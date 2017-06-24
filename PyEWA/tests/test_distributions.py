import numpy as np
import numpy.testing as npt

from unittest import TestCase

from PyEWA.distributions import Distribution


class TestUpdatePdf(object):
    def test_value(self):

        d = Distribution()
        assert np.all(d.pdf==1)
        assert d.pdf.shape == (10, )
