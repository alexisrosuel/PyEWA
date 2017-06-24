
"""
EWA.prior
A collection of common prior distributions
"""

class Uniform:
    def __init__(self, lower=0, upper=1):
        self.lower = lower
        self.upper = upper

        self.density = 1 / (self.upper - self.lower)

    def pdf(x):
        return self.density if (x > self.lower and x < self.upper) else 0
