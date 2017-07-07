
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
        self.support = support

    def update_pdf(self, pdf):
        self.pdf = pdf

    def plot_distribution(self):
        if len(self.support.shape) == 1:
            plt.scatter(x=self.support, y=self.pdf)
            plt.xlabel('$\\theta$')
            plt.ylabel('$f(\\theta)$')
            plt.title('Distribution')
            plt.grid(True)
            plt.show()

        elif len(self.support.shape) == 2:
            extent = [self.support.min(), self.support.max(),
                      self.support.min(), self.support.max()]

            plt.imshow(self.pdf, cmap='Reds',
                       interpolation='nearest', extent=extent, origin='lower')

            plt.show()

        else:
            print('impossible to print, dimension of input > 2')


class Uniform:
    def __init__(self, support=np.linspace(0, 1, 10)):
        self.support = support
        self.densite = 1. / np.prod(self.support.shape)
        self.pdf = (self.support * 0) + self.densite
