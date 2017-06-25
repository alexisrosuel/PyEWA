import numpy as np

def bernstein_function(x):
    return (np.exp(x) - 1 - x) / (x ** 2)
