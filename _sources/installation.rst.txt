Installation
============

.. note:: Importantly, note that this codebase is written for Python 3 and is not backwards compatible.


Installation through pip
------------------------

You can install delfi by cloning from the github repository and using pip:

.. code-block:: console

    git clone https://github.com/mackelab/delfi.git
    cd delfi
    pip install -r requirements.txt
    pip install .

Note that depending on how your system is set up, you may need to type ``pip3`` instead of ``pip``. 
A comfortable way for developing is using ``pip install -e .`` instead of the last line above; see `setuptools documentation`_ for more information.

.. _the code repository: https://github.com/mackelab/delfi
.. _setuptools documentation: http://setuptools.readthedocs.io/en/latest/setuptools.html#develop-deploy-the-project-source-in-development-mode

How to update
-------------

Pull from upstream. Re-execute the install command, in case you used ``setup.py install``. This step is not necessary if you used ``setup.py develop``, since the package directory is linked symbolically.


Requirements
------------

Installing delfi as described above will automatically take care of the requirements.

The installation requirements are specified in setup.py. Core dependencies are theano and lasagne. For lasagne, delfi relies on the development version of lasagne (0.2dev) rather than the stable version (0.1) that is available through pip.

To use APT with Gaussian or Mixture-of-Gaussians proposals, you will likely need to make openblas available to theano. You can do this on many linux systems using ``sudo apt-get install libopenblas-dev``
