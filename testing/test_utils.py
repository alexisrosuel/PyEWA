import numpy as np
import numpy.testing as npt

from unittest import TestCase

from pyewa.utils import create_support


class TestUtils(object):
    def test_create_support(self):
        support = create_support()
        npt.assert_allclose(support, np.linspace(0, 1, 10))

        support = create_support(
            lower=10, upper=15, density=5, base_dimension=4)
        assert len(support.shape) == 4
        npt.assert_allclose(support.shape, np.array([5, 5, 5, 5]))
