__doc__ = """...
"""

import numpy as np

from .loss_functions import Squared
from .distributions import Distribution, Uniform
from .bases import Constant, Linear
from .utils import bernstein_function, create_support


class EWA:
    """class Exponentially weighted aggregation
    Class for Exponentially weighted aggregation
    Dependencies:
        numpy
        scipy
        matplotlib
    Inputs:
        ...

    Callable Methods:

    References:
        ...
    """

    def __init__(self, learning_rate='auto', output_dimension=1, input_dimension=1, base_dimension=1, B=1, base='constant', loss_function='squared',
                 prior='uniform', lower=0, upper=1, density=5):
        self.learning_rate = learning_rate

        # The bound of the data, assume all |X| < B
        self.B = B

        # The upper and lower bound of the Y
        self.lower = lower
        self.upper = upper
        self.density = density
        self.step = float(upper - lower) / density

        # create the support of the distributions: hypercube in input_dimension dimensions, with density point for each axes
        self.base_dimension = base_dimension
        self.support = create_support(
            lower=self.lower, upper=self.upper, density=self.density, base_dimension=self.base_dimension)

        self.output_dimension = output_dimension
        self.input_dimension = input_dimension

        self.base = base
        self.loss_function = loss_function
        self.prior = prior
        self.create_parameters()

        # initialize distribution with the prior
        self.distribution = Distribution(
            support=self.support, pdf=self.prior.pdf)

        # we saw 0 examples for the moment
        self.n = 0

    def create_parameters(self):
        if self.base == 'constant':
            self.base = Constant(support=self.support,
                                 output_dimension=self.output_dimension)
        elif self.base == 'linear':
            self.base = Linear(support=self.support,
                               output_dimension=self.output_dimension)

        if self.prior == 'uniform':
            self.prior = Uniform(support=self.support)

        if self.loss_function == 'squared':
            self.loss_function = Squared(B=self.B)

    def fit(self, X, Y):
        """ For multiple X and y at one time
        X : np array, shape=(number of examples, dimension of the input)
        Y : np array, shape=(number of examples, dimension of the prediction)

        learning_rate optimal : 2 sqrt(2n log(M))/C
        """
        nb_examples, _ = Y.shape

        # if learning_rate is adaptative, compute it
        auto = False
        if self.learning_rate == 'auto':
            self.learning_rate = 2 * \
                np.sqrt(2 * nb_examples *
                        np.log(self.base.M)) / self.loss_function.C
            auto = True

        loss_by_example = np.array([self.loss_function.loss(
            Y[i], self.base.evaluate(X[i])) for i in range(nb_examples)])

        W = np.exp(-self.learning_rate *
                   np.sum(loss_by_example, axis=0)) * \
            self.prior.pdf

        self.distribution.pdf = (W / np.sum(W)) / \
            (self.step ** self.input_dimension)

        self.update_prior()
        self.n += X.shape[0]

        # reset learning_rate to 'auto'
        if auto:
            self.learning_rate = 'auto'

    def update_prior(self):
        self.prior = self.distribution

    def predict(self, x):
        """ input : x array of shape nb of dimension of the output_dimension
        output : array of shape dimension of the output """
        all_predictions = self.base.evaluate(x)
        prediction_weighted_by_pdf = np.array(
            [all_predictions[index] * self.distribution.pdf[index] for index, _ in np.ndenumerate(self.distribution.pdf)])

        return np.sum(prediction_weighted_by_pdf, axis=0) * self.step ** self.input_dimension

    def weak_bound_regret(self, epsilon=0.05):
        """ with probability as least 1-epsilon, this bound for the regret is true
        non optimal learning rate : lambda C^2 / (4n) + 2 (log M + log 2/epsilon) / lamnda
        optimal learning rate : C sqrt(2 log(M) / n) + C log(2/epsilon) / sqrt(2n log(M))"""
        if self.learning_rate == 'auto':
            return self.loss_function.C * (np.sqrt(2 * np.log(self.base.M) / self.n) + np.log(2 / epsilon) / np.sqrt(2 * self.n * np.log(self.base.M)))
        else:
            return self.learning_rate * self.loss_function.C ** 2 / (4 * self.n) + 2 * (np.log(self.base.M) + np.log(2 / epsilon)) / self.learning_rate

    """
    def strong_bound_regret(self, epsilon):
        ratio = (1 + (lam * c)/n * g(2*lam*c/n)) / (1 - (lam * c)/n * g(2*lam*c/n))
        KLdiv =
    """
