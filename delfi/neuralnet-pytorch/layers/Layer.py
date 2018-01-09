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

    def add_param(self, spec, shape, name, **kwargs):
        shape = tuple([ int(x) for x in shape ])
        init = spec(shape)
        data = nn.Parameter(dtype(init))
        param = { 'data' : data,'init' : init, 'shape' : shape, 'name' : name, **kwargs }
        self.params[name] =param
        self.register_parameter(name, data)
        return data

    def get_params(self, **tags):
        ret = []

        for k in self.params:
            add = True
            for t in tags:
                if tags[t] and ((not t in self.params[k]) or not self.params[k][t]):
                    add=False
                    break
                elif not tags[t] and ((t in self.params[k]) and self.params[k][t]):   
                    add=False
                    break

            if add:
                ret.append(self.params[k]['data'])

        return ret
         

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


class Initialiser:
    def __call__(self, shape):
        return self.sample(shape)

class Normal(Initialiser):
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def sample(self, shape):
        ms = torch.zeros(*shape).type(dtype) + self.mean
        return torch.normal(ms, self.std).type(dtype)

class He(Initialiser):
    def __init__(self, initialiser, gain=1.0, c01b=False):
        self.initialiser = initialiser
        self.gain = gain
        self.c01b = c01b

    def sample(self, shape):
        if self.c01b:
            raise NotImplementedError
        else:
            if len(shape) == 2:
                fan_in = shape[0]
            elif len(shape) > 2:
                fan_in = np.prod(shape[1:])
            else:
                raise RuntimeError("This initializer only works with shapes of length >= 2")

        std = self.gain * np.sqrt(1.0/fan_in)
        return self.initialiser(std=std).sample(shape)

class HeNormal(He):
    def __init__(self, gain=1.0, c01b=False):
        super().__init__(Normal, gain, c01b)

class Constant(Initialiser):
    def __init__(self, val):
        self.val = val

    def sample(self, shape):
        return torch.ones(*shape).type(dtype)
