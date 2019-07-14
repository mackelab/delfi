``distribution`` – Distributions
--------------------------------

Several probability distributions are implemented in the module ``delfi.distribution``.

The general interface for distributions is specified in the abstract class
``BaseDistribution.py``: Each distribution needs to implement the abstract
methods  and properties of this class. Mixture distributions and the respective base class are in the module ``delfi.distribution.mixture``.

Discrete
````````
.. autoclass:: delfi.distribution.Discrete
  :show-inheritance:
  :inherited-members:
  :members:

Gamma
`````
.. autoclass:: delfi.distribution.Gamma
  :show-inheritance:
  :inherited-members:
  :members:

Gaussian
````````
.. autoclass:: delfi.distribution.Gaussian
  :show-inheritance:
  :inherited-members:
  :members:

Student's T
```````````
.. autoclass:: delfi.distribution.StudentsT
  :show-inheritance:
  :inherited-members:
  :members:

Transformed Distribution
````````````````````````
.. autoclass:: delfi.distribution.TransformedDistribution
  :show-inheritance:
  :inherited-members:
  :members:

Uniform
```````
.. autoclass:: delfi.distribution.Uniform
  :show-inheritance:
  :inherited-members:
  :members:
