PyEWA
=======

Exponentially Weigthed Aggregation for Python

.. image:: https://codecov.io/gh/alexisrosuel/PyEWA/badge.svg
    :target: https://codecov.io/gh/alexisrosuel/PyEWA/

.. image:: https://travis-ci.org/alexisrosuel/PyEWA.svg?branch=master
    :target: https://travis-ci.org/alexisrosuel/PyEWA

.. image:: https://ci.appveyor.com/api/projects/status/github/alexisrosuel/PyEWA?branch=master&svg=true
    :target: https://ci.appveyor.com/project/alexisrosuel/PyEWA

.. image:: https://circleci.com/gh/alexisrosuel/PyEWA/tree/master.svg?style=shield&circle-token=:circle-token
    :target: https://circleci.com/gh/alexisrosuel/PyEWA

The code supports exponentialweighted aggregation (ewa for short). Some standard base functions (constant, linear, etc) are built-in. 

Examples of the use of this package are shown below. 

The kriging methods are separated into four classes. 

PyEWA will later be on PyPi, so installation is as simple as typing the following into a command line.

.. code:: bash

    pip install pyewa

To update PyEWA from PyPi, type the following into a command line.

.. code:: bash

    pip install --upgrade pyewa

PyKrige uses the MIT-Clause License.

Exponentially weighted aggregation example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from pyewa.ewa import EWA
    import numpy as np
    
    data = np.array([[0.3, 1.2, 0.47],
                     [1.9, 0.6, 0.56],
                     [1.1, 3.2, 0.74],
                     [3.3, 4.4, 1.47],
                     [4.7, 3.8, 1.74]])

References
^^^^^^^^^^
PAC-Bayesian bounds for the Exponentially Weighted Aggregate (EWA) in the online setting. Slow rates, fast rates. Examples: classification, regression. Multiplicative weights algorithms for the MS-type aggregation.

N. Cesa-Bianchi & G. Lugosi, Prediction, learning and games, Cambridge University Press, 2006.

S. Gerchinovitz, Prediction of individual sequences and prediction in the statistical framework: some links around sparse regression and aggregation techniques, PhD Thesis, Univ. Paris 11, 2011. (Chapters 2 and 3).

Hoeffding and Bernstein inequalities. PAC-Bayesian bounds for the EWA in the batch setting. Slow rates in the general case. Fast rates under Bernstein and margin assumptions. Examples: classification, regression, matrix factorization.

O. Catoni, Pac-Bayesian supervised classification: the thermodynamics of statistical learning, IMS Lecture Notes, 2007.
