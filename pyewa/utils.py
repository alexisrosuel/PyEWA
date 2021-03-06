import numpy as np


def create_support(lower=0, upper=1, density=10, base_dimension=1):
    obj = np.linspace(start=lower, stop=upper, num=density)
    support = obj
    for i in range(base_dimension - 1):
        support = np.array([support for i in range(density)])

    return support
