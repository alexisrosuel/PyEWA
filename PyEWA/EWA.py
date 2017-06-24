__doc__ = """...
"""

import numpy as np

from loss_functions import Squared
from distributions import Distribution, Uniform
from bases import Constant


class EWA:
    """class OnlineLearning
    Class for online learning
    Dependencies:
        numpy
        scipy
        matplotlib
    Inputs:
        X (array-like): Coordinates of data points.
        Y (array-like): Y-coordinates of data points.
        Z (array-like): Values at data points.

    Callable Methods:

    References:
        ...
    """

    def __init__(self, base='constant', loss_function='squared', learning_rate=0.1, prior='uniform'):
        self.support = np.linspace(0, 1, 10)

        self.base = base
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.prior = prior

        self.create_parameters()

        # initialize distribution with the prior
        self.distribution = Distribution(support=self.support, pdf=self.prior.pdf(self.support))

    def create_parameters(self):
        if self.base == 'constant':
            self.base = Constant()

        if self.loss_function == 'squared':
            self.loss_function = Squared()

        if self.prior == 'uniform':
            self.prior = Uniform()

    def fit(self, X, y):
        """ For multiple X and y at one time """
        W = np.exp(-self.learning_rate *
                   self.loss_function.loss(y, self.base.evaluate(x)) *
                   self.prior.pdf(x))
        return 0

    def update_distribution(self, x, y):
        W = np.exp(-self.learning_rate *
                   self.loss_function.loss(y, self.base.evaluate(x)) *
                   self.distribution.pdf)

        self.distribution.update_pdf(W / np.sum(W))

    def update_prior(self):
        self.prior = self.distribution

    def predict(self, x):
        return np.sum(self.base.evaluate(x) * self.distribution)
