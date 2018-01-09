import lasagne
import lasagne.init as linit
import lasagne.layers as ll
import lasagne.nonlinearities as lnl
import theano
import theano.tensor as tt

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

dtype = theano.config.floatX


class FullyConnectedLayer(lasagne.layers.Layer):
    def __init__(self, incoming, n_units, svi=True,
                 mW_init=linit.HeNormal(), mb_init=linit.Constant([0.]),
                 sW_init=linit.Constant([-5.]), sb_init=linit.Constant([-5.]),
                 actfun=lnl.tanh, seed=None, **kwargs):
        """Fully connected layer with optional weight uncertainty

        Parameters
        ----------
        incoming : lasagne.layers.Layer instance
            Incoming layer
        n_units : int
            Number of units
        svi : bool
            Weight uncertainty
        mW_init : function
            Function to initialise weights for mean of weight (multiplicative)
        mb_init : function
            Function to initialise weights for mean of weight (bias)
        sW_init : function
            Function to initialise weights for log std of weight (multiplicative)
        sb_init : function
            Function to initialise weights for log std of weight (bias)
        actfun : function
            Activation function
        """
        super(FullyConnectedLayer, self).__init__(incoming, **kwargs)
        self.n_units = n_units
        self.actfun = actfun
        self.svi = svi

        self.mW = self.add_param(mW_init,
                                 (self.input_shape[1], self.n_units),
                                 name='mW', mp=True, wp=True)
        self.mb = self.add_param(mb_init,
                                 (self.n_units,),
                                 name='mb', mp=True, bp=True)

        if self.svi:
            self._srng = RandomStreams(
                lasagne.random.get_rng().randint(
                    1, 2147462579))
            self.sW = self.add_param(sW_init,
                                     (self.input_shape[1], self.n_units),
                                     name='sW', sp=True, wp=True)
            self.sb = self.add_param(sb_init,
                                     (self.n_units,),
                                     name='sb', sp=True, bp=True)

    def get_output_for(self, input, deterministic=False, **kwargs):
        """Returns matrix with shape (batch, n_units)"""
        ma = tt.dot(input, self.mW) + self.mb
        if not self.svi or deterministic:
            return self.actfun(ma)
        else:
            sa = tt.dot(input**2, tt.exp(2 * self.sW)) + tt.exp(2 * self.sb)
            ua = self._srng.normal((input.shape[0], self.n_units), dtype=dtype)
            return self.actfun(tt.sqrt(sa) * ua + ma)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.n_units)
