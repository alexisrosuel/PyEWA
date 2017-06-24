
"""
EWA.prior
A collection of common distributions
"""

from scipy.stats import uniform
import numpy as np

class Distribution:
    def __init__(self, support=np.linspace(0, 1, 10), pdf=np.ones(shape=10)):
        self.pdf = pdf

    def update_pdf(self, pdf):
        self.pdf = pdf

    def plot_distribution(self):
        return 0

class Uniform:
    def __init__(self, lower=0, upper=1):
        self.lower = lower
        self.upper = upper

    def pdf(self, x):
        return uniform.pdf(x, loc=self.lower, scale=self.upper-self.lower)
