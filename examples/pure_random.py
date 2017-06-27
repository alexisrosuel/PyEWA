import sys
sys.path.insert(0, '.')

#
# Demonstrates the usage of EWA
# See ... for more details
#

import numpy as np

from pyewa.ewa import EWA
from scipy.stats import uniform, norm


def get_data():
    X = uniform.rvs(loc=10, scale=1, size=100)
    X = np.array([[xi, xi] for xi in X])
    #X = np.array([[X[2 * i], X[2 * i + 1]] for i in range(50)])
    Y = norm.rvs(loc=0, scale=1, size=100)
    Y = np.array([[yi, yi] for yi in Y])
    return (X, Y)


def apply_EWA():
    data = get_data()
    X = data[0]
    Y = data[1]

    input_dimension = X.shape[1]
    output_dimension = Y.shape[1]
    ewa = EWA(input_dimension=input_dimension, output_dimension=output_dimension, base_dimension=2,
              learning_rate='auto', density=50, B=11, lower=-1, upper=1)

    ewa.fit(X, Y)
    """print(ewa.distribution.pdf[5, 5, 5])
    print(ewa.distribution.pdf[0, 9, 0])"""
    """print(ewa.weak_bound_regret(0.01))
    print(ewa.weak_bound_regret(0.5))
    print(ewa.weak_bound_regret(0.99))"""
    ewa.distribution.plot_distribution()

    """print(ewa.predict(X[5]))"""


if __name__ == '__main__':
    apply_EWA()
