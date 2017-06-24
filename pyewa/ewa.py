__doc__ = """...
"""

import numpy as np

from .loss_functions import Squared
from .distributions import Distribution, Uniform
from .bases import Constant


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

    def __init__(self, base='constant', loss_function='squared', learning_rate=0.1, prior='uniform', support=np.linspace(0, 1, 10)):
        self.support = support
        # Compute the step between two evaluation of the distribution
        self.step = (self.support.max() - self.support.min()) / self.support.shape[0]
        self.base = base
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.prior = prior

        self.create_parameters()

        # initialize distribution with the prior
        self.distribution = Distribution(support=self.support, pdf=self.prior.pdf)

    def create_parameters(self):
        if self.base == 'constant':
            self.base = Constant(support=self.support)

        if self.prior == 'uniform':
            self.prior = Uniform(support=self.support)

        if self.loss_function == 'squared':
            self.loss_function = Squared()

    def fit(self, X, Y):
        """ For multiple X and y at one time
        X : np array, shape=(number of examples, dimension of the problem)
        Y : np array, shape=number of examples
        """
        loss_by_example = np.array([self.loss_function.loss(y, self.base.evaluate(x)) for x, y in zip(X, Y)])
        W = np.exp(-self.learning_rate *
                   np.sum(loss_by_example, axis=0)) * \
            self.prior.pdf

        self.distribution.pdf = (W / np.sum(W)) / self.step

    def update_distribution(self, x, y):
        """
        x: np array of shape the number of dimension of the problem
        y: int
        """
        W = np.exp(-self.learning_rate *
                   self.loss_function.loss(y, self.base.evaluate(x))) * \
            self.distribution.pdf

        self.distribution.pdf = (W / np.sum(W)) / self.step

    def update_prior(self):
        self.prior = self.distribution

    def predict(self, x):
        return np.sum(self.base.evaluate(x) * self.distribution.pdf)
