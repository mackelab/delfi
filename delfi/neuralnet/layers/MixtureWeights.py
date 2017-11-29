import lasagne
import lasagne.init as linit
import lasagne.layers as ll
import lasagne.nonlinearities as lnl
import theano
import theano.tensor as tt

from delfi.neuralnet.layers.FullyConnected import FullyConnectedLayer

dtype = theano.config.floatX


class MixtureWeightsLayer(FullyConnectedLayer):
    def __init__(self, incoming, n_units, svi=True,
                 mW_init=linit.HeNormal(), mb_init=linit.Constant([0.]),
                 sW_init=linit.Constant([-5.]), sb_init=linit.Constant([-5.]),
                 actfun=lnl.softmax, **kwargs):
        """Mixture weights layer with optional weight uncertainty

        If n_units > 1, this becomes a fully-connected layer. Else, no
        parameters are added, and the output defaults to weight 1.

        See ``delfi.neuralnet.layers.FullyConnected`` for docstring
        """
        self.n_units = n_units

        if n_units > 1:
            super(MixtureWeightsLayer, self).__init__(
                incoming,
                n_units,
                svi=svi,
                mW_init=mW_init,
                mb_init=mb_init,
                sW_init=sW_init,
                sb_init=sb_init,
                actfun=actfun,
                **kwargs)
        else:
            # init of lasagne.layers.Layer
            super(FullyConnectedLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, deterministic=False, **kwargs):
        """Returns matrix with shape (batch, n_units)"""
        if self.n_units > 1:
            return super(MixtureWeightsLayer, self).get_output_for(
                input,
                deterministic=deterministic,
                **kwargs)
        else:
            return tt.ones((input.shape[0], self.n_units), dtype=dtype)
