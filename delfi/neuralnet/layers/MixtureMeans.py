import lasagne
import lasagne.init as linit
import lasagne.layers as ll
import lasagne.nonlinearities as lnl
import theano
import theano.tensor as tt

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

dtype = theano.config.floatX


class MixtureMeansLayer(lasagne.layers.Layer):
    def __init__(self,
                 incoming,
                 n_components,
                 n_dim,
                 svi=True,
                 mWs_init=linit.HeNormal(),
                 mbs_init=linit.Normal(1.),
                 sWs_init=linit.Constant([-5.]),
                 sbs_init=linit.Constant([-5.]),
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
            self._srng = RandomStreams(
                lasagne.random.get_rng().randint(
                    1, 2147462579))
            self.sWs = [self.add_param(sWs_init,
                                       (self.input_shape[1], self.n_dim),
                                       name='sW' + str(c), sp=True, wp=True)
                        for c in range(n_components)]
            self.sbs = [self.add_param(sbs_init,
                                       (self.n_dim,),
                                       name='sb' + str(c), sp=True, bp=True)
                        for c in range(n_components)]

    def get_output_for(self, input, deterministic=False, **kwargs):
        """Compute outputs

        Returns
        -------
        list of length n_components with (batch, n_dim)
        """
        if not self.svi or deterministic:
            return [
                tt.dot(
                    input,
                    mW) + mb for mW,
                mb in zip(
                    self.mWs,
                    self.mbs)]
        else:
            uas = [
                self._srng.normal(
                    (input.shape[0],
                     self.n_dim),
                    dtype=dtype) for i in range(
                    self.n_components)]
            mas = [
                tt.dot(
                    input,
                    mWm) +
                mbm for mWm,
                mbm in zip(
                    self.mWs,
                    self.mbs)]
            sas = [tt.dot(input**2, tt.exp(2 * sW)) + tt.exp(2 * sb)
                   for sW, sb in zip(self.sWs, self.sbs)]
            zas = [tt.sqrt(sa) * ua + ma for sa, ua, ma in zip(sas, uas, mas)]
            return zas

    def get_output_shape_for(self, input_shape):
        raise NotImplementedError
