__doc__ = """...
"""

import numpy as np

from .loss_functions import Squared
from .distributions import Distribution, Uniform
from .bases import Constant
from .utils import bernstein_function


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

    def __init__(self, learning_rate=0.1, B=1, base='constant', loss_function='squared', prior='uniform', support=np.linspace(0, 1, 10)):
        self.learning_rate = learning_rate
        # The bound of the data, assume all |X| < B
        self.B = B

        self.support = support

        # Compute the step between two evaluation of the distribution
        self.step = (self.support.max() - self.support.min()) / self.support.shape[0]
        self.base = base
        self.loss_function = loss_function

        self.prior = prior

        self.create_parameters()

        # initialize distribution with the prior
        self.distribution = Distribution(support=self.support, pdf=self.prior.pdf)

        # we saw 0 examples for the moment
        self.n = 0

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

        self.update_prior()
        self.n += X.shape[0]

    def update_distribution(self, x, y):
        """
        x: np array of shape the number of dimension of the problem
        y: int
        """
        W = np.exp(-self.learning_rate *
                   self.loss_function.loss(y, self.base.evaluate(x))) * \
            self.distribution.pdf

        self.distribution.pdf = (W / np.sum(W)) / self.step

        self.update_prior()
        self.n += 1

    def update_prior(self):
        self.prior = self.distribution

    def predict(self, x):
        return np.sum(self.base.evaluate(x) * self.distribution.pdf)

    def weak_bound_regret(self, epsilon):
        """ with probability as least 1-epsilon, this bound for the regret is true """
        M = self.support.shape[0]
        return np.sqrt(2 * np.log(M) / self.n) + np.log(2/epsilon) / np.sqrt(2 * self.n * np.log(M))

    """
    def strong_bound_regret(self, epsilon):
        ratio = (1 + (lam * c)/n * g(2*lam*c/n)) / (1 - (lam * c)/n * g(2*lam*c/n))
        KLdiv =
    """
