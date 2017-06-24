from __future__ import absolute_import
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

#
# Demonstrates the usage of online learning
# See ... for more details
#

from EWA import EWA
from scipy.stats import uniform

def get_data():
    X = uniform.rvs(loc=0, scale=1, size=10)
    y = 2 * X
    return (X, y)

def apply_EWA():
    data = get_data()
    X = data[0]
    y = data[1]

    ewa = EWA()
    ewa.update_distribution(X[0], y[0])
    print(ewa.distribution.pdf)

if __name__ == '__main__':
    apply_EWA()
