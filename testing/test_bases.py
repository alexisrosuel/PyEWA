import numpy as np
import numpy.testing as npt

from unittest import TestCase

from pyewa.bases import Constant, Linear
from pyewa.utils import create_support


class TestConstant(object):
    def test_constant_init(self):
        c = Constant()
        assert c.M == 10

        # Test multidimensionnal support
        support = create_support(base_dimension=3)
        c = Constant(support=support, output_dimension=3)
        assert c.M == 10 ** 3

    def test_f(self):
        c = Constant()
        theta = 1
        assert c.f(theta, 0) == theta

        support = create_support(base_dimension=3)
        c = Constant(support=support, output_dimension=3)
        theta = [1, 2, 3]
        npt.assert_allclose(c.f(theta, 10), theta)

    def test_evaluate(self):
        support = create_support()
        c = Constant(support=support)
        npt.assert_allclose(c.evaluate(x=1), [[x]
                                              for x in support], rtol=1e-5, atol=1e-5)

        support = create_support(base_dimension=3)
        c = Constant(support=support, output_dimension=3)
        assert c.evaluate(10).shape == (10, 10, 10, 3)


class TestLinear(object):
    def test_constant_init(self):
        support = create_support(base_dimension=3)
        l = Linear(support=support, input_dimension=3, output_dimension=1)
        assert l.M == 10

    """    # Test multidimensionnal support
        support = create_support(base_dimension=3)
        c = Linear(support=support, output_dimension=3)
        assert c.M == 10 ** 3

    def test_f(self):
        c = Constant()
        theta = 1
        assert c.f(theta, 0) == theta

        support = create_support(base_dimension=3)
        c = Constant(support=support, output_dimension=3)
        theta = [1, 2, 3]
        npt.assert_allclose(c.f(theta, 10), theta)

    def test_evaluate(self):
        support = create_support()
        c = Constant(support=support)
        npt.assert_allclose(c.evaluate(x=1), [[x]
                                              for x in support], rtol=1e-5, atol=1e-5)

        support = create_support(base_dimension=3)
        c = Constant(support=support, output_dimension=3)
        assert c.evaluate(10).shape == (10, 10, 10, 3)
"""
