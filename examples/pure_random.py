import sys
sys.path.insert(0, '.')

#
# Demonstrates the usage of EWA
# See ... for more details
#

import numpy as np

from pyewa.ewa import EWA
from scipy.stats import uniform

def get_data():
    X = uniform.rvs(loc=0, scale=1, size=50)
    y = np.array([[Xi, Xi] for Xi in X])
    X = np.array([[Xi] for Xi in X])
    return (X, y)

def apply_EWA():
    data = get_data()
    X = data[0]
    Y = data[1]

    ewa = EWA(support=np.linspace(start=0, stop=1, num=1000), output_dimension=2, learning_rate='auto')
    # ewa.update_distribution(X[0], Y[0])
    ewa.fit(X,Y)
    print(ewa.weak_bound_regret(0.01))
    print(ewa.weak_bound_regret(0.5))
    print(ewa.weak_bound_regret(0.99))
    ewa.distribution.plot_distribution()

    print(ewa.predict(X[5]))

if __name__ == '__main__':
    apply_EWA()
