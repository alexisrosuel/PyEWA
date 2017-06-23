__doc__ = """...
"""


class OnlineLearning:
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

    def __init__(self, X, y, pi, functions, learning_rate, loss_function):
        self.support = np.linspace(0, 1, 10)

        self.X = X
        self.y = y
        self.pi = pi
        self.functions = functions
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.distribution = [self.pi(theta) for theta in self.support] # initialize new distribution with pi

    def update_distribution(self, X, y):
        Wt = np.sum([np.exp(-self.learning_rate * self.loss_function(y, f[theta](X))) * self.distribution(theta) for theta in self.support])
        self.distribution = [np.exp(-self.learning_rate * self.loss_function(y, f[theta](X))) * self.distribution(theta) for theta in self.support] / Wt

    def predict(self, x):
        return np.sum([functions[theta](x) * self.pi(theta) for theta in self.support])
