Installation
============

.. note:: Importantly, note that this codebase is written for Python 3 and is not backwards compatible.


Installation through pip
------------------------

delfi is `hosted on PyPI`_, the Python software repository.

You can install delfi by

.. code-block:: console

    pip install delfi --user --process-dependency-links

Note that depending on how your system is set up, you may need to type ``pip3`` instead of ``pip``.

.. _hosted on PyPI: https://pypi.python.org/pypi/delfi


Manual installation
-------------------

Alternatively, you can clone `the code repository`_ and run the setup:

.. code-block:: console

    git clone https://github.com/mackelab/delfi.git
    cd delfi
    python setup.py install --user

Note that a comfortable way for developing is using ``python setup.py develop --user`` instead of the last line above; see `setuptools documentation`_ for more information.

.. _the code repository: https://github.com/mackelab/delfi
.. _setuptools documentation: http://setuptools.readthedocs.io/en/latest/setuptools.html#develop-deploy-the-project-source-in-development-mode

How to update
-------------

If you installed using pip, you can update the package using:

.. code-block:: console

    pip install delfi --upgrade --upgrade-strategy only-if-needed --user --process-dependency-links

If you installed by cloning the repository, pull from upstream. Re-execute the install command, in case you used ``setup.py install``. This step is not necessary if you used ``setup.py develop``, since the package directory is linked symbolically.


Requirements
------------

Installing delfi as described above, will automatically take care of the requirements.

The installation requirements are specified in setup.py. Core dependencies are theano and lasagne. For lasagne, delfi relies on the development version of lasagne (0.2dev) rather than the stable version (0.1) that is available through pip.
