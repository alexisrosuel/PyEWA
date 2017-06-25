import numpy as np
import numpy.testing as npt

from unittest import TestCase

from pyewa.bases import Constant


class TestConstant(object):
    def test_init(self):
        c = Constant()
        assert c.support.shape == (10, )
        npt.assert_allclose(c.support, np.linspace(0, 1, 10))

        c = Constant(support=np.linspace(2, 10, 15))
        assert c.support.shape == (15, )
        npt.assert_allclose(c.support, np.linspace(2, 10, 15))

    def test_evaluate(self):
        c = Constant()
        npt.assert_allclose(c.evaluate(x=1), [[x] for x in np.linspace(0, 1, 10)])
