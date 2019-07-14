``simulator`` – Simulator models
--------------------------------

Forward models, which can be written in any programming language, need to be
wrapped in class that inherits from ``delfi.simulator.SimulatorBase``. The
base class defines the interface.

The following simulators are currently part of ``delfi`` for testing and to
provide examples:

Blob
````
.. autoclass:: delfi.simulator.Blob
  :show-inheritance:
  :inherited-members:
  :members:

Gauss
`````
.. autoclass:: delfi.simulator.Gauss
  :show-inheritance:
  :inherited-members:
  :members:

Gaussian Mixture
````````````````
.. autoclass:: delfi.simulator.GaussMixture
  :show-inheritance:
  :inherited-members:
  :members:

Transformed Simulator
`````````````````````
.. autoclass:: delfi.simulator.TransformedSimulator
  :show-inheritance:
  :inherited-members:
  :members:

TwoMoons
````````
.. autoclass:: delfi.simulator.TwoMoons
  :show-inheritance:
  :inherited-members:
  :members: