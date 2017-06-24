import numpy as np
import numpy.testing as npt

from unittest import TestCase

from pyewa.distributions import Distribution, Uniform


class TestDistributions(object):
    def test_init(self):
        d = Distribution()
        assert np.all(d.pdf==1)
        assert d.pdf.shape == (10, )
        npt.assert_allclose(d.support, np.linspace(0, 1, 10))

        d = Distribution(support=np.linspace(2, 10, 15), pdf=np.zeros(shape=15))
        assert np.all(d.pdf==0)
        assert d.pdf.shape == (15, )
        npt.assert_allclose(d.support, np.linspace(2, 10, 15))

    def test_update_pdf(self):
        d = Distribution()
        d.update_pdf(pdf=np.zeros(shape=10))
        assert np.all(d.pdf==0)


class TestUniform(object):
    def test_init(self):
        u =  Uniform()
        assert u.lower == 0
        assert u.upper == 1
        assert np.all(u.pdf==1)
