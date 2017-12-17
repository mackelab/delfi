import torch
import torch.nn as nn

from delfi.neuralnet.layers.Layer import Layer

dtype = torch.DoubleTensor

class MixturePrecisionsLayer(Layer):
    def __init__(self,
                 incoming,
                 n_components,
                 n_dim,
                 mWs_init=linit.HeNormal(),
                 mbs_init=linit.Constant([0.]),
                 **kwargs):
        """Fully connected layer for mixture precisions, optional weight uncertainty

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
        super(MixturePrecisionsLayer, self).__init__(incoming, **kwargs)
        self.n_components = n_components
        self.n_dim = n_dim

        self.mWs = [self.add_param(mWs_init,
                                   (self.input_shape[1], self.n_dim**2),
                                   name='mW' + str(c), mp=True, wp=True)
                    for c in range(n_components)]
        self.mbs = [self.add_param(mbs_init,
                                   (self.n_dim**2,),
                                   name='mb' + str(c), mp=True, bp=True)
                    for c in range(n_components)]

    def forward(self, inp, **kwargs):
        """Compute outputs

        Returns
        -------
        list by concatenation of
            Us : list of length n_components with (batch, n_dim, n_dim)
                Precision factors
            ldetUs : list of length n_components with (batch, n_dim, n_dim)
                Log determinants of precisions
        """
        triu_mask = np.triu(np.ones([self.n_dim, self.n_dim], dtype=dtype), 1)
        diag_mask = np.eye(self.n_dim, dtype=dtype)
        offdiag_mask = np.ones(self.n_dim, dtype=dtype) - \
            np.eye(self.n_dim, dtype=dtype)

            zas_reshaped = [(torch.mm(
                inp, mW).view + mb).view((-1, self.n_dim, self.n_dim)) for mW, mb in zip(self.mWs, self.mbs)]

        Us = [
            triu_mask *
            za +
            diag_mask *
            torch.exp(
                diag_mask *
                za) for za in zas_reshaped]
        ldetUs = [torch.sum(torch.sum(diag_mask * za, dim=2), dim=1)
                  for za in zas_reshaped]

        return {'Us': Us, 'ldetUs': ldetUs}
