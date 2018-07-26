import lasagne
import lasagne.init as linit
import lasagne.layers as ll
import lasagne.nonlinearities as lnl
import numpy as np
import theano
import theano.tensor as tt

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

dtype = theano.config.floatX


class MixturePrecisionsLayer(lasagne.layers.Layer):
    def __init__(self,
                 incoming,
                 n_components,
                 n_dim,
                 svi=True,
                 rank=None,
                 homoscedastic=False,
                 mWs_init=linit.HeNormal(),
                 mbs_init=linit.Constant([0.]),
                 sWs_init=linit.Constant([-5.]),
                 sbs_init=linit.Constant([-5.]),
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
        self.rank = rank
        assert not homoscedastic
        self.homoscedastic = homoscedastic
        self.n_dim = n_dim
        self.svi = svi

        self.mWs = [self.add_param(mWs_init,
                                   (self.input_shape[1], self.n_dim**2),
                                   name='mW' + str(c), mp=True, wp=True)
                    for c in range(n_components)]
        self.mbs = [self.add_param(mbs_init,
                                   (self.n_dim**2,),
                                   name='mb' + str(c), mp=True, bp=True)
                    for c in range(n_components)]

        if self.svi:
            self._srng = RandomStreams(
                lasagne.random.get_rng().randint(
                    1, 2147462579))
            self.sWs = [self.add_param(sWs_init,
                                       (self.input_shape[1], self.n_dim**2),
                                       name='sW' + str(c), sp=True, wp=True)
                        for c in range(n_components)]
            self.sbs = [self.add_param(sbs_init,
                                       (self.n_dim**2,),
                                       name='sb' + str(c), sp=True, bp=True)
                        for c in range(n_components)]

    def get_output_for(self, input, deterministic=False, **kwargs):
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
        if not self.rank is None:
            triu_mask[self.rank:] *= 0.
        diag_mask = np.eye(self.n_dim, dtype=dtype)
        offdiag_mask = np.ones(self.n_dim, dtype=dtype) - \
            np.eye(self.n_dim, dtype=dtype)

        if not self.svi or deterministic:
            zas_reshaped = [tt.reshape(tt.dot(input, mW) + mb, 
                [-1, self.n_dim, self.n_dim]) for mW, mb in zip(self.mWs, self.mbs)]
        else:
            uas = [
                self._srng.normal(
                    (input.shape[0],
                     self.n_dim**2),
                    dtype=dtype) for i in range(
                    self.n_components)]
            mas = [tt.dot(input,mW) + mb for mW, mb in zip(self.mWs,self.mbs)]
            sas = [tt.dot(input**2, tt.exp(2 * sW)) + tt.exp(2 * sb)
                   for sW, sb in zip(self.sWs, self.sbs)]
            zas = [tt.sqrt(sa) * ua + ma for sa, ua, ma in zip(sas, uas, mas)]
            zas_reshaped = [tt.reshape(
                za, [-1, self.n_dim, self.n_dim]) for za in zas]

        Us = [
            triu_mask *
            za +
            diag_mask *
            tt.exp(
                diag_mask *
                za) for za in zas_reshaped]
        ldetUs = [tt.sum(tt.sum(diag_mask * za, axis=2), axis=1)
                  for za in zas_reshaped]

        return {'Us': Us, 'ldetUs': ldetUs}

    def get_output_shape_for(self, input_shape):
        raise NotImplementedError


class MixtureHomoscedasticPrecisionsLayer(lasagne.layers.Layer):
    def __init__(self,
                 incoming,
                 n_components,
                 n_dim,
                 svi=True,
                 rank=None,
                 homoscedastic=True,
                 mbs_init=linit.Constant([0.]),
                 sbs_init=linit.Constant([-5.]),
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
        super(MixtureHomoscedasticPrecisionsLayer, self).__init__(incoming, **kwargs)
        self.n_components = n_components
        self.rank = rank
        assert homoscedastic
        self.homoscedastic = homoscedastic
        self.n_dim = n_dim
        self.svi = svi

        #self.mWs = [self.add_param(mWs_init,
        #                           (self.input_shape[1], self.n_dim**2),
        #                           name='mW' + str(c), mp=True, wp=True)
        #            for c in range(n_components)]
        self.mbs = [self.add_param(mbs_init,
                                   (self.n_dim**2,),
                                   name='mb' + str(c), mp=True, bp=True)
                    for c in range(n_components)]

        if self.svi:
            self._srng = RandomStreams(
                lasagne.random.get_rng().randint(
                    1, 2147462579))
            #self.sWs = [self.add_param(sWs_init,
            #                           (self.input_shape[1], self.n_dim**2),
            #                           name='sW' + str(c), sp=True, wp=True)
            #            for c in range(n_components)]
            self.sbs = [self.add_param(sbs_init,
                                       (self.n_dim**2,),
                                       name='sb' + str(c), sp=True, bp=True)
                        for c in range(n_components)]

    def get_output_for(self, input, deterministic=False, **kwargs):
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
        if not self.rank is None:
            triu_mask[self.rank:] *= 0.
        diag_mask = np.eye(self.n_dim, dtype=dtype)
        offdiag_mask = np.ones(self.n_dim, dtype=dtype) - \
            np.eye(self.n_dim, dtype=dtype)

        if not self.svi or deterministic:
            zas_reshaped = [tt.reshape(mb + 0.*tt.sum(input) , 
                [-1, self.n_dim, self.n_dim]) for mb in self.mbs]
        else:
            uas = [
                self._srng.normal(
                    (input.shape[0],
                     self.n_dim**2),
                    dtype=dtype) for i in range(
                    self.n_components)]
            mas = [ mb + 0.*tt.sum(input) for mb in self.mbs ]
            sas = [ tt.exp(2 * sb) for sb in self.sbs]
            zas = [tt.sqrt(sa) * ua + ma for sa, ua, ma in zip(sas, uas, mas)]
            zas_reshaped = [tt.reshape(
                za, [-1, self.n_dim, self.n_dim]) for za in zas]

        Us = [
            triu_mask *
            za +
            diag_mask *
            tt.exp(
                diag_mask *
                za) for za in zas_reshaped]
        ldetUs = [tt.sum(tt.sum(diag_mask * za, axis=2), axis=1)
                  for za in zas_reshaped]

        return {'Us': Us, 'ldetUs': ldetUs}

    def get_output_shape_for(self, input_shape):
        raise NotImplementedError
