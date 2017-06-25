PyEWA
=======

Exponentially Weigthed Aggregation for Python

.. image:: https://travis-ci.org/alexisrosuel/PyEWA.svg?branch=master
    :target: https://travis-ci.org/alexisrosuel/PyEWA

.. image:: https://ci.appveyor.com/api/projects/status/github/bsmurphy/PyKrige?branch=master&svg=true
    :target: https://ci.appveyor.com/project/bsmurphy/pykrige

.. image:: https://circleci.com/gh/bsmurphy/PyKrige/tree/master.svg?style=shield&circle-token=:circle-token
    :target: https://circleci.com/gh/bsmurphy/PyKrige

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
