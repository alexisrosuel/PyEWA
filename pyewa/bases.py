
"""
EWA.bases
A collection of common functions bases
"""

import numpy as np

class Constant:
    def __init__(self, support=np.linspace(0, 1, 10), output_dimension=1):
        self.output_dimension = output_dimension
        self.support = support

    def f(self, x, theta):
        """ f_theta(x) = theta
        input : x array of shape dimension of the input
        output : array of shape dimension of the prediction"""
        return theta * np.ones(shape=self.output_dimension)

    def evaluate(self, x):
        """ input : array of dimension dimension of the input
        Returns a vector of shape ('dimension of the support, dimension of the prediction')
        """
        return np.array([self.f(x, theta) for theta in self.support])
