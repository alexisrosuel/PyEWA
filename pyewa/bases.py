
"""
EWA.bases
A collection of common functions bases
"""

import numpy as np

class Constant:
    def __init__(self, support=np.linspace(0, 1, 10)):
        self.support = support

    def evaluate(self, x):
        return np.array([theta for theta in self.support])
