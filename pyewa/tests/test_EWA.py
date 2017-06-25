import numpy as np
import numpy.testing as npt

from unittest import TestCase, mock

from pyewa.ewa import EWA
from pyewa.bases import Constant
from pyewa.loss_functions import Squared
from pyewa.distributions import Uniform, Distribution


class TestEWA(object):
    def test_init(self):
        ewa = EWA()
        npt.assert_allclose(ewa.support, np.linspace(0, 1, 10))
        assert ewa.step == 0.1
        assert ewa.learning_rate == 0.1
        assert ewa.support.shape == (10, )
        print(ewa.prior)
        npt.assert_allclose(ewa.prior.support, Uniform().support)
        npt.assert_allclose(ewa.prior.pdf, Uniform().pdf)
        npt.assert_allclose(ewa.base.support, Constant().support)
        npt.assert_allclose(ewa.distribution.pdf, Distribution().pdf)

    def test_fit(self):
        ewa = EWA()
        X = np.array([[0.5], [0.7], [0.3]])
        Y = np.array([[0.5], [0.4], [0.3]])
        ewa.fit(X=X, Y=Y)
        npt.assert_allclose(ewa.distribution.pdf, np.array([0.985128, 1.008011, 1.023814, 1.03219 , 1.032955, 1.026092, 1.011751,
                                                            0.990249, 0.962051, 0.927759]), atol=1e-4)

    def test_update_prior(self):
        ewa = EWA()
        ewa.distribution.pdf = [2, 3, 4]
        ewa.update_prior()
        npt.assert_allclose(ewa.prior.pdf, [2, 3, 4])

    @mock.patch('pyewa.ewa.Constant.evaluate', return_value=[[1, 2, 3], [1, 2, 3]])
    def test_predict(self, evaluate_faked):
        ewa = EWA(output_dimension=3, support=np.array([0, 1]))
        print(ewa.predict(x=[2]))
        print(ewa.distribution.pdf)
        npt.assert_allclose(ewa.predict(x=2), np.array([1, 2, 3]))
