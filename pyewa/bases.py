
"""
EWA.bases
A collection of common functions bases
"""

import numpy as np


class Constant:
    def __init__(self, support=np.linspace(0, 1, 10), output_dimension=1):
        """
        For this class, the parameter theta must have the same dimension as the output_dimension.
        """
        if len(support.shape) != output_dimension:
            raise ValueError(
                'Impossible to use constant base with dimension theta != dimension output')

        self.output_dimension = output_dimension
        self.support = support
        # total number of functions
        self.M = np.prod(self.support.shape)

    def f(self, theta, x):
        """ f_theta(x) = theta
        input : x array of shape dimension of the input
        output : array of shape dimension of the prediction"""
        return theta * np.ones(shape=self.output_dimension)

    def evaluate(self, x):
        """ input : x: array of dimension dimension of the input
        Returns a vector of shape ('dimension of the support, dimension of the prediction')
        """
        min_support = self.support.min()
        max_support = self.support.max()
        len_support = len(self.support[..., :])

        result_1d = np.array([self.f(min_support + np.array(theta) * (max_support - min_support) / (len_support - 1), x)
                              for theta, _ in np.ndenumerate(self.support)])

        newshape = self.support.shape + (self.output_dimension,)
        result = np.reshape(result_1d, newshape=newshape)
        return result


class Linear:
    def __init__(self, support=np.linspace(0, 1, 10), output_dimension=1, input_dimension=1):
        """
        For this class, the parameter theta must have the dimension (input_dimension, output_dimension). As it was initially
        a vector of dimension 'base_dimension', it must be reshaped to 'output_dimension x input_dimension'
        """
        # Try to reshape theta
        if len(support.shape) != output_dimension * input_dimension:
            raise ValueError(
                'Impossible to use linear base with dimension theta != (dimension input, dimension_output)')

        # if we can, we reshape theta
        self.support = np.reshape(support, newshape=(
            output_dimension, input_dimension))

        self.output_dimension = output_dimension
        self.support = support
        # total number of functions
        self.M = np.prod(self.support.shape)

    def f(self, theta, x):
        """ f_theta(x) = theta . x
        input : x array of shape dimension of the input
        output : array of shape dimension of the prediction"""
        return np.dot(theta, x)

    def evaluate(self, x):
        """ input : x: array of dimension dimension of the input
        Returns a vector of shape ('dimension of the support, dimension of the prediction')
        """
        min_support = self.support.min()
        max_support = self.support.max()
        len_support = len(self.support[..., :])

        result_1d = np.array([self.f(min_support + np.array(theta) * (max_support - min_support) / (len_support - 1), x)
                              for theta, _ in np.ndenumerate(self.support)])
        newshape = self.support.shape + (self.output_dimension,)
        result = np.reshape(result_1d, newshape=newshape)
        return result

# class Quadratic : f_theta(x) = xT.theta.x
