import torch
import torch.nn as nn

import numpy as np

from delfi.neuralnet.layers.Layer import Layer

dtype = torch.DoubleTensor

class ReplaceMissingLayer(Layer):
    def __init__(self, incoming, n_inputs=None, **kwargs):
        """Inputs that are NaN will be replaced by zero through this layer"""
        super(ReplaceMissingLayer, self).__init__(incoming, **kwargs)
        self.output_shape = self.input_shape

    def forward(self, inp, **kwargs):
        ret = inp.clone()
        ret[ret != ret] = 0
        return ret.type(dtype)

