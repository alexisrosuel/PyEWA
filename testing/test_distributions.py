import numpy as np
import numpy.testing as npt

from unittest import TestCase

from pyewa.distributions import Distribution, Uniform
from pyewa.utils import create_support


"""class TestDistributions(object):
    def test_update_pdf(self):
        d = Distribution()
        d.update_pdf(pdf=np.zeros(shape=10))
        assert np.all(d.pdf == 0)"""


class TestUniform(object):
    def test_init(self):
        support = create_support()
        u = Uniform(support)
        assert np.all(u.pdf == 0.1)

        support = create_support(
            lower=0, upper=2, density=20, base_dimension=3)
        u = Uniform(support=support)
        assert np.all(u.pdf == 1.25e-4)
