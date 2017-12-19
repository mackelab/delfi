import torch
import torch.nn as nn

import numpy as np

from delfi.neuralnet.layers.Layer import *

dtype = torch.DoubleTensor

class ImputeMissingLayer(Layer):
    def __init__(self, incoming, n_inputs, R=Normal(0.01), **kwargs):
        """Inputs that are NaN will be replaced by zero through this layer"""
        super().__init__(incoming, **kwargs)
        self.output_shape = self.input_shape
        self.R = self.add_param(R, (*n_inputs,), name='imputation_values')

    def forward(self, inp, **kwargs):
        ret = inp.clone()
        ret[ret != ret] = self.R[ret != ret]
        return ret.type(dtype)

class ReplaceMissingLayer(Layer):
    def __init__(self, incoming, n_inputs=None, **kwargs):
        """Inputs that are NaN will be replaced by zero through this layer"""
        super().__init__(incoming, **kwargs)
        self.output_shape = self.input_shape

    def forward(self, inp, **kwargs):
        ret = inp.clone()
        ret[ret != ret] = 0
        return ret.type(dtype)

