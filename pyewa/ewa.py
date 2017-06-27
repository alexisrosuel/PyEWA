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


<< << << < 744e6e8cf13ce119a5a9522ae6042b74314e0fd5
    def __init__(self, learning_rate='auto', output_dimension=1, B=1, base='constant', loss_function='squared',
                 prior='uniform', support=np.linspace(0, 1, 10)):
== == == =
    def __init__(self, learning_rate='auto', output_dimension=1, input_dimension=1, B=1, base='constant', loss_function='squared',
                 prior='uniform', lower=0, upper=1, density=5):
>>>>>> > Wip: need to separate the dimension of the input and the dimension of the parameter we want to learn
        self.learning_rate = learning_rate
        # The bound of the data, assume all |X| < B
        self.B = B

        self.lower = lower
        self.upper = upper
        self.density = density
        self.step = float(upper - lower) / density
        self.output_dimension = output_dimension
        self.input_dimension = input_dimension
        # create the support of the distributions: cube in input_dimension dimensions, with density point for each axes
        self.create_support()

<< << << < 744e6e8cf13ce119a5a9522ae6042b74314e0fd5
        self.support = support

        # Compute the step between two evaluation of the distribution
        self.step = (self.support.max() - self.support.min()) / \
            self.support.shape[0]
== == == =
>>>>>> > Wip: need to separate the dimension of the input and the dimension of the parameter we want to learn
        self.base = base
        self.loss_function = loss_function
        self.prior = prior
        # Compute the step between two evaluation of the distribution
        self.create_parameters()

        # initialize distribution with the prior
        self.distribution = Distribution(
            support=self.support, pdf=self.prior.pdf)

        # we saw 0 examples for the moment
        self.n = 0

    def create_support(self):
        obj = np.linspace(start=self.lower, stop=self.upper, num=self.density)
        support = obj
        for i in range(self.input_dimension - 1):
            support = np.array([support for i in range(self.density)])
        self.support = support

    def create_parameters(self):
        if self.base == 'constant':
            self.base = Constant(support=self.support,
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
        prediction_weighted_by_pdf = np.einsum(
            'ij,i->ij', self.base.evaluate(x), self.distribution.pdf)
        return np.sum(prediction_weighted_by_pdf, axis=0) * self.step

    def weak_bound_regret(self, epsilon):
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
