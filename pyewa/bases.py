
"""
EWA.bases
A collection of common functions bases
"""

import numpy as np


class Constant:
    def __init__(self, support=np.linspace(0, 1, 10), output_dimension=1):
        self.output_dimension = output_dimension
        self.support = support
        # total number of functions
        self.M = np.prod(self.support.shape)

    def f(self, theta, x):
        """ f_theta(x) = theta
        input : x array of shape dimension of the input
        output : array of shape dimension of the prediction"""
        return np.sum(theta) * np.ones(shape=self.output_dimension)

    def evaluate(self, x):
        """ input : x: array of dimension dimension of the input
        Returns a vector of shape ('dimension of the support, dimension of the prediction')
        """
        min_support = self.support.min()
        max_support = self.support.max()
        len_support = len(self.support[..., :])

        result_1d = np.array([self.f(min_support + np.array(theta) * (max_support - min_support) / len_support, x)
                              for theta, _ in np.ndenumerate(self.support)])
        newshape = self.support.shape + (self.output_dimension,)
        result = np.reshape(result_1d, newshape=newshape)
        return result
