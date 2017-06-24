#
# Demonstrates the usage of online learning
# See ... for more details
#

import Exponentially_Weighted_Aggregation.EWA
from scipy.stats import uniform

def get_data():
    X = uniform.rvs(loc=0, scale=1, size=10, random_state=32)
    y = 2 * X
    return (X, y)

def apply_EWA():
    data = get_data()
    X = data[0]
    y = data[1]

    EWA.EWA()
    EWA.update_distribution(X[0], y[0])
    print(EWA.distribution)

if __name__ == '__main__':
    apply_EWA()
