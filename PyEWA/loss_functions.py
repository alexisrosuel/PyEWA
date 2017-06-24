
"""
EWA.loss_functions
A collection of common loss functions
"""

class Squared:
    def __init__(self):
        self.rien = 0

    def loss(self, y1, Y):
        # y1 : one number
        # Y : numpy array
        return (y1 - Y) ** 2
