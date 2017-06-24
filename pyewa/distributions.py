
"""
EWA.prior
A collection of common distributions
"""

from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt


class Distribution:
    def __init__(self, support=np.linspace(0, 1, 10), pdf=np.ones(shape=10)):
        self.pdf = pdf
        self.support=support

    def update_pdf(self, pdf):
        self.pdf = pdf

    def plot_distribution(self):
        plt.scatter(x=self.support, y=self.pdf)
        plt.xlabel('$\\theta$')
        plt.ylabel('$f(\\theta)$')
        plt.title('Distribution')
        plt.grid(True)
        plt.show()

class Uniform:
    def __init__(self, support=np.linspace(0, 1, 10)):
        self.support = support
        self.lower = self.support.min()
        self.upper = self.support.max()

        self.pdf = uniform.pdf(self.support, loc=self.lower, scale=self.upper-self.lower)
