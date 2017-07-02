import sys
sys.path.insert(0, '.')

#
# Demonstrates the usage of EWA
# See ... for more details
#

import numpy as np
import matplotlib.pyplot as plt

from pyewa.ewa import EWA
from scipy.stats import uniform, norm


def get_data_1d():
    N = 25
    X = uniform.rvs(loc=10, scale=1, size=N)
    X = np.array([[xi] for xi in X])
    Y = norm.rvs(loc=10, scale=1, size=N)
    Y = np.array([[yi] for yi in Y])
    return (X, Y)


def get_data_2d():
    N = 25
    X = uniform.rvs(loc=0, scale=1, size=N)
    X = np.array([[xi, xi] for xi in X])
    Y = np.array([np.cos(xi) for xi in X])
    return (X, Y)


def get_data_3d():
    N = 25
    X = uniform.rvs(loc=0, scale=1, size=N)
    X = np.array([[xi] for xi in X])
    Y = np.array([np.cos(xi) for xi in X])
    return (X, Y)


def apply_EWA():
    # Test constant 1d
    X, Y = get_data_1d()

    input_dimension = X.shape[1]
    output_dimension = Y.shape[1]

    lower = Y.min()
    upper = Y.max()
    B = np.max(np.abs(X))

    ewa = EWA(input_dimension=input_dimension, output_dimension=output_dimension, base_dimension=1,
              base='constant', learning_rate='auto', density=80, B=B, lower=lower, upper=upper)
    ewa.fit(X, Y)
    ewa.distribution.plot_distribution()
    plt.scatter(X.ravel(), [ewa.predict(x)
                            for x in X.ravel()], marker='o', color='r')
    plt.scatter(X.ravel(), Y.ravel(), marker='o', color='b')
    plt.show()

    # Test constant 2d
    X, Y = get_data_2d()
    print(Y)

    input_dimension = X.shape[1]
    output_dimension = Y.shape[1]

    lower = Y.min()
    upper = Y.max()
    B = np.max(np.abs(X))

    ewa = EWA(input_dimension=input_dimension, output_dimension=output_dimension, base_dimension=2,
              base='constant', learning_rate='auto', density=80, B=B, lower=lower, upper=upper)
    ewa.fit(X, Y)
    ewa.distribution.plot_distribution()
    plt.scatter(X.ravel(), [ewa.predict(x)
                            for x in X.ravel()], marker='o', color='r')
    plt.scatter(X.ravel(), Y.ravel(), marker='o', color='b')
    plt.show()

    # Test constant 3d

    # Test linear 1d

    # Test linear 2d

    # Test linear 3d


if __name__ == '__main__':
    apply_EWA()
