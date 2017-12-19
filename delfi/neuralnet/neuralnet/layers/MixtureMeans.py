import torch
import torch.nn as nn

from delfi.neuralnet.layers.Layer import Layer

dtype = torch.DoubleTensor


class MixtureMeansLayer(Layer):
    def __init__(self,
                 incoming,
                 n_components,
                 n_dim,
                 mWs_init=None,
                 mbs_init=None,
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

        self.mWs = [self.add_param(mWs_init,
                                   (self.input_shape[1], self.n_dim),
                                   name='mW' + str(c), mp=True, wp=True)
                    for c in range(n_components)]
        self.mbs = [self.add_param(mbs_init,
                                   (self.n_dim,),
                                   name='mb' + str(c), mp=True, bp=True)
                    for c in range(n_components)]

    def forward(self, inp, **kwargs):
        """Compute outputs

        Returns
        -------
        list of length n_components with (batch, n_dim)
        """
        return [
            torch.mm(
                inp,
                mW) + mb for mW,
            mb in zip(
                self.mWs,
                self.mbs)]
