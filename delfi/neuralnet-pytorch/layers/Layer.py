import collections
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

dtype = torch.DoubleTensor

class Layer(nn.Module):
    def __init__(self, incoming, **kwargs):
        super().__init__()
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.kwargs = kwargs
        self.params = collections.OrderedDict()

    def add_param(self, init, shape, name, **kwargs):
        s = 0.3
        temp = np.random.normal(scale=s, size=shape)
        data = nn.Parameter(dtype(temp))
        param = { 'data' : data,'init' : init, 'shape' : shape, 'name' : name, **kwargs }
        self.params[name] =param
        self.register_parameter(name, data)
        return data

    def forward(self, inp, **kwargs):
        raise NotImplementedError

class FlattenLayer(Layer):
    def __init__(self, incoming, outdim, **kwargs):
        super().__init__(incoming, **kwargs)
        self.outdim = outdim
        to_flatten = self.input_shape[self.outdim - 1:]
        self.output_shape = self.input_shape[:self.outdim - 1] + (np.prod(to_flatten),)

    def forward(self, inp):
        args = [ inp.shape[0] ] + [ int(x) for x in self.output_shape[1:]]
        ret = inp.view(*args)
        return ret

class ReshapeLayer(Layer):
    def __init__(self, incoming, output_shape, **kwargs):
        super().__init__(incoming, **kwargs)
        self.output_shape = output_shape

    def forward(self, inp):
        ret = inp.view(*self.output_shape)
        return ret
