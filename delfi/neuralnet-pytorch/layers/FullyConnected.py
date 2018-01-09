import torch
import torch.nn as nn
import torch.nn.functional as F

from delfi.neuralnet.layers.Layer import *

dtype = torch.DoubleTensor

class FullyConnectedLayer(Layer):
    def __init__(self, incoming, n_units, svi=False,
                 mWs_init=HeNormal(), mbs_init=Constant([0.]),
                 sWs_init=Constant([-5.]), sbs_init=Constant([-5.]),
                 actfun=F.tanh, seed=None, **kwargs):
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
        self.svi = svi

        # Init mW, mb, determine input shapes
        self.mW = self.add_param(mWs_init,
                                 (self.input_shape[1], self.n_units),
                                 name='mW', mp=True, wp=True)
        self.mb = self.add_param(mbs_init,
                                 (self.n_units,),
                                 name='mb', mp=True, bp=True)

        self.output_shape = (self.input_shape[0], self.n_units)

        if self.svi:
            if seed == None:
                seed = np.random.randint(1, 2147462579)
            self._srng = np.random.RandomState(seed)
            self.sW = self.add_param(sWs_init,
                                     (self.input_shape[1], self.n_units),
                                     name='sW', sp=True, wp=True)
            self.sb = self.add_param(sbs_init,
                                     (self.n_units,),
                                     name='sb', sp=True, bp=True)

    def forward(self, inp, deterministic=False, **kwargs):
        """Returns matrix with shape (batch, n_units)"""
        ma = torch.mm(inp, self.mW) + self.mb
        if not self.svi or deterministic:
            return self.actfun(ma)
        else:
            sa = torch.mm(inp**2, torch.exp(2 * self.sW)) + torch.exp(2 * self.sb)
            ua = Variable(dtype(self._srng.normal(size=(inp.shape[0], self.n_units))))
            return self.actfun(torch.sqrt(sa) * ua + ma)
