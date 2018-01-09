import torch
import torch.nn as nn

from delfi.neuralnet.layers.Layer import *

dtype = torch.DoubleTensor


class MixtureMeansLayer(Layer):
    def __init__(self,
                 incoming,
                 n_components,
                 n_dim,
                 svi=True,
                 mWs_init=HeNormal(),
                 mbs_init=Constant([0.]),
                 sWs_init=Constant([-5.]),
                 sbs_init=Constant([-5.]),
                 seed=None,
                 **kwargs):
        """Fully connected layer for mixture means, optional weight uncertainty

        Parameters
        ----------
        incoming : lasagne.layers.Layer instance
            Incoming layer
        n_components : int
            Number of components
        n_dim : int
            Dimensionality of output vector
        svi : bool
            Weight uncertainty
        mWs_init : function
            Function to initialise weights for mean of weight (multiplicative);
            applied per component
        mbs_init : function
            Function to initialise weights for mean of weight (bias);
            applied per component
        sWs_init : function
            Function to initialise weights for log std of weight (multiplicative);
            applied per component
        sbs_init : function
            Function to initialise weights for log std of weight (bias);
            applied per component
        """
        super(MixtureMeansLayer, self).__init__(incoming, **kwargs)
        self.n_components = n_components
        self.n_dim = n_dim
        self.svi = svi

        self.mWs = [self.add_param(mWs_init,
                                   (self.input_shape[1], self.n_dim),
                                   name='mW' + str(c), mp=True, wp=True)
                    for c in range(n_components)]
        self.mbs = [self.add_param(mbs_init,
                                   (self.n_dim,),
                                   name='mb' + str(c), mp=True, bp=True)
                    for c in range(n_components)]

        if self.svi:
            if seed == None:
                seed = np.random.randint(1, 2147462579)
            self._srng = np.random.RandomState(seed)
            self.sWs = [self.add_param(sWs_init,
                                       (self.input_shape[1], self.n_dim),
                                       name='sW' + str(c), sp=True, wp=True)
                        for c in range(n_components)]
            self.sbs = [self.add_param(sbs_init,
                                       (self.n_dim,),
                                       name='sb' + str(c), sp=True, bp=True)
                        for c in range(n_components)]

    def forward(self, inp, deterministic=False, **kwargs):
        """Compute outputs

        Returns
        -------
        list of length n_components with (batch, n_dim)
        """
        if not self.svi or deterministic:
            return [
                torch.mm(
                    inp,
                    mW) + mb for mW,
                mb in zip(
                    self.mWs,
                    self.mbs)]
        else:
            uas = [
                Variable(dtype(self._srng.normal(size=\
                    (inp.shape[0],
                     self.n_dim)))) for i in range(
                    self.n_components)]
            mas = [
                torch.mm(
                    inp,
                    mWm) +
                mbm for mWm,
                mbm in zip(
                    self.mWs,
                    self.mbs)]
            sas = [torch.mm(inp**2, torch.exp(2 * sW)) + torch.exp(2 * sb)
                   for sW, sb in zip(self.sWs, self.sbs)]
            zas = [torch.sqrt(sa) * ua + ma for sa, ua, ma in zip(sas, uas, mas)]
            return zas
