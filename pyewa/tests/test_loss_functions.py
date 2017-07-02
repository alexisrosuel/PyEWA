import numpy as np
import numpy.testing as npt

from unittest import TestCase, mock

from pyewa.loss_functions import Squared


class TestSquared(object):
    def test_init(self):
        squared = Squared(B=1)
        assert squared.C == 4
        assert squared.max_learning_rate == 1 / 8

    def test_loss(self):
        squared = Squared(B=1)

        # 1d
        y1 = 4
        Y = np.array([10, 5])
        npt.assert_allclose(squared.loss(y1, Y), 37.0)

        # 2d
        y1 = np.array([10, 3])
        Y = np.array([[10, 5], [2, 1]])
        npt.assert_allclose(squared.loss(y1, Y), [4.0, 68.0])

        # 2d
        y1 = np.array([15, 2, 4])
        Y = np.array([[15, 2, 4], [15, 2, 4], [15, 2, 4]])
        print(squared.loss(y1, Y))
        npt.assert_allclose(squared.loss(y1, Y), [0, 0, 0])
