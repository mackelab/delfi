import torch
import torch.nn as nn
import torch.nn.functional as F

from delfi.neuralnet.layers.FullyConnected import FullyConnectedLayer

dtype = torch.DoubleTensor

class MixtureWeightsLayer(FullyConnectedLayer):
    def __init__(self, incoming, n_units, 
                 mW_init=None, mb_init=None,
                 actfun=F.softmax, **kwargs):
        """Mixture weights layer with optional weight uncertainty

        If n_units > 1, this becomes a fully-connected layer. Else, no
        parameters are added, and the output defaults to weight 1.

        See ``delfi.neuralnet.layers.FullyConnected`` for docstring
        """
        self.n_units = n_units

        if n_units > 1:
            super(MixtureWeightsLayer, self).__init__(
                incoming,
                n_units,
                mW_init=mW_init,
                mb_init=mb_init,
                actfun=actfun,
                **kwargs)
        else:
            super().__init__(incoming, 1, actfun=actfun, **kwargs)

    def forward(self, inp, **kwargs):
        """Returns matrix with shape (batch, n_units)"""
        if self.n_units > 1:
            return super().forward(inp, **kwargs)
        else:
            return torch.ones((inp.shape[0], 1)).type(dtype)
