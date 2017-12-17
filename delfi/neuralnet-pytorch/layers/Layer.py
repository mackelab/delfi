import collections
import torch
import torch.nn as nn

dtype = torch.DoubleTensor

class Layer(nn.Module):
    def __init__(self, incoming, **kwargs):
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.kwargs = kwargs
        self.params = collections.OrderedDict()

    def add_param(self, init, shape, name='', **kwargs):
        param = { 'init' : init, 'shape' : shape, 'name' : name, **kwargs }
        self.params.add(param)

    def forward(self, inp, **kwargs):
        raise NotImplementedError

class FlattenLayer(Layer):
    def __init__(self, incoming, outdim, **kwargs):
        super().__init__(incoming, kwargs)
        self.outdim = outdim
        to_flatten = self.input_shape[self.outdim - 1:]
        self.output_shape = self.input_shape[:self.outdim - 1] + (np.prod(to_flatten.shape),)

    def forward(self, inp):
        raise NotImplementedError("Todo")

class InputLayer(Layer):
    def __init__(self, incoming, input_var, **kwargs):
        super().__init__(incoming, **kwargs)
        self.input_var = input_var
        self.output_shape = self.input_shape

    
    def forward(self, inp):
        return inp
