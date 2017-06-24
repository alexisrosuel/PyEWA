__doc__ = """...
"""

from loss_functions import Squared
from priors import Uniform
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

    def __init__(self, base=Constant, loss_function=Squared, learning_rate=0.1, prior=Uniform):
        self.support = np.linspace(0, 1, 10)

        self.X = X
        self.y = y
        self.base = base
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.prior = prior

        self.distribution = [self.pi(theta) for theta in self.support] # initialize new distribution with pi

    def fit(self, X, y):
        """ For multiple X and y at one time """
        return 0

    def update_distribution(self, x, y):
        W = np.exp(-self.learning_rate *
                   self.loss_function.loss(y, self.base.evaluate(x))) *
                   self.distribution)

        self.distribution = W / np.sum(W)

    def predict(self, x):
        return np.sum(self.base.evaluate(x) * self.distribution)

    def plot_distribution(self):
        return 0
