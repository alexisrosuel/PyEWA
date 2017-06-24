
"""
EWA.bases
A collection of common functions bases
"""

class Constant:
    def __init__(self, lower=0, upper=1, step=0.1, support=np.linspace(0, 1, 10)):
        self.lower = lower
        self.upper = upper
        self.step = step
        self.support = support

    def f(x):
        return x

    def evaluate(x):
        return np.array([f(x) for x in self.support])
