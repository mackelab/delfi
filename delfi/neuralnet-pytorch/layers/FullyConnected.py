import torch
import torch.nn as nn
import torch.nn.functional as F

from delfi.neuralnet.layers.Layer import Layer

dtype = torch.DoubleTensor

class FullyConnectedLayer(Layer):
    def __init__(self, incoming, n_units,
                 actfun=F.tanh, **kwargs):
        """Fully connected layer with optional weight uncertainty

        Parameters
        ----------
        incoming : layer
            Incoming layer
        n_units : int
            Number of units
        actfun : function
            Activation function
        """
        super(FullyConnectedLayer, self).__init__(incoming, **kwargs)
        self.n_units = n_units
        self.actfun = actfun

        # Init mW, mb, determine input shapes
        self.mW = self.add_param(mW_init,
                                 (self.input_shape[1], self.n_units),
                                 name='mW', mp=True, wp=True)
        self.mb = self.add_param(mb_init,
                                 (self.n_units,),
                                 name='mb', mp=True, bp=True)

        self.output_shape = (self.input_shape[0], self.n_units)

    def forward(self, inp, **kwargs):
        """Returns matrix with shape (batch, n_units)"""
        ma = torch.mm(inp, self.mW) + self.mb
        return self.actfun(ma)
