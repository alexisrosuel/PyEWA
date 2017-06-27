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


def get_data():
    N = 25
    X = uniform.rvs(loc=10, scale=1, size=N)
    X = np.array([[xi] for xi in X])
    # X = np.array([[X[2 * i], X[2 * i + 1]] for i in range(50)])
    Y = norm.rvs(loc=10, scale=1, size=N)
    Y = np.array([[yi] for yi in Y])
    return (X, Y)


def get_data_2():
    N = 50
    X = uniform.rvs(loc=10, scale=1, size=N)
    X = np.array([[xi, xi] for xi in X])
    # X = np.array([[X[2 * i], X[2 * i + 1]] for i in range(50)])
    Y = norm.rvs(loc=0, scale=1, size=N)
    Y = np.array([[yi, yi] for yi in Y])
    return (X, Y)


def get_data_3():
    N = 25
    X = uniform.rvs(loc=0, scale=1, size=N)
    X = np.array([[xi] for xi in X])

    Y = np.array([np.cos(xi) for xi in X])
    return (X, Y)


def apply_EWA():
    data = get_data_3()
    X = data[0]
    Y = data[1]

    input_dimension = X.shape[1]
    output_dimension = Y.shape[1]

    lower = Y.min()
    upper = Y.max()

    ewa = EWA(input_dimension=input_dimension, output_dimension=output_dimension, base_dimension=2,
              base='linear', learning_rate='auto', density=80, B=2, lower=-20, upper=20)
    ewa.fit(X, Y)

    print(ewa.distribution.pdf)
    ewa.distribution.plot_distribution()
    print(X.ravel())
    plt.scatter(X.ravel(), [ewa.predict(x)
                            for x in X.ravel()], marker='o', color='r')
    plt.scatter(X.ravel(), Y.ravel(), marker='o', color='b')

    plt.show()


if __name__ == '__main__':
    apply_EWA()
