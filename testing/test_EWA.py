import numpy as np
import numpy.testing as npt

from unittest import TestCase, mock

from pyewa.ewa import EWA
from pyewa.bases import Constant
from pyewa.loss_functions import Squared
from pyewa.distributions import Uniform, Distribution
from pyewa.utils import create_support


"""class TestEWA(object):
    def test_fit(self):
        # Test in dimension 1
        ewa = EWA(base='constant', input_dimension=1, output_dimension=1)
        X = np.array([[0.5], [0.7], [0.3]])
        Y = np.array([[0.5], [0.4], [0.3]])
        ewa.fit(X=X, Y=Y)
        print(ewa.distribution.pdf)
        npt.assert_allclose(ewa.distribution.pdf, np.array(
            [0.76982,  1.461298,  1.548971,  0.916859,  0.303052]), atol=1e-4)

        # Test in dimension 2
        # plus tard : @mock.patch pyewa.ewa.create_support(return_value=create_support())
        ewa = EWA(input_dimension=2, base_dimension=2, base='linear')
        X = np.array([[0.5, 0.8], [0.7, 0.1], [0.3, 0.4]])
        Y = np.array([[0.5], [0.4], [0.3]])
        npt.assert_allclose(ewa.distribution.pdf, np.array([0.985128, 1.008011, 1.023814, 1.03219, 1.032955, 1.026092, 1.011751,
                                                            0.990249, 0.962051, 0.927759]), atol=1e-4)

    def test_update_prior(self):
        ewa = EWA()
        ewa.distribution.pdf = [2, 3, 4]
        ewa.update_prior()
        npt.assert_allclose(ewa.prior.pdf, [2, 3, 4])

    @mock.patch('pyewa.ewa.Constant.evaluate', return_value=[[1, 2, 3], [1, 2, 3]])
    def test_predict(self, evaluate_faked):
        ewa = EWA(output_dimension=2, base_dimension=2)
        print(ewa.predict(x=[2]))
        print(ewa.distribution.pdf)
        npt.assert_allclose(ewa.predict(x=2), np.array([1, 2, 3]))

    def test_weak_bound_regret(self):
        ewa = EWA(learning_rate='auto')
        ewa.n = 10
        npt.assert_allclose(ewa.weak_bound_regret(), 4.8701778727)
        npt.assert_allclose(ewa.weak_bound_regret(epsilon=0.5), 3.24678524847)

        ewa = EWA(learning_rate=0.1)
        ewa.n = 10
        npt.assert_allclose(ewa.weak_bound_regret(
            epsilon=0.5), 59.954645471079814)"""
