
"""
EWA.loss_functions
A collection of common loss functions. They have to be convex
"""

import numpy as np

class Squared:
    def __init__(self, B=1):
        # Bound for the loss function, assumes for all y, y', |l(y,y')| < C
        self.C = 4 * (B ** 2)
        # Bound for the learning rate in order to guarantee exp-convexity of the loss function
        self.max_learning_rate = 1 / (8 * B ** 2)

    def loss(self, y1, Y):
        """y1 : array of dimension dimension of the prediction
        Y : array of dimension (dimension of support, dimension of the prediction) """
        return np.linalg.norm(y1 - Y, axis=1) ** 2
