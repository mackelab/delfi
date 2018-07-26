import collections
import delfi.distribution as dd
import delfi.neuralnet.layers as dl
import lasagne
import lasagne.layers as ll
import lasagne.nonlinearities as lnl
import numpy as np
import theano
import theano.tensor as tt

from delfi.utils.odict import first, last, nth

dtype = theano.config.floatX

def MyLogSumExp(x, axis=None):
    x_max = tt.max(x, axis=axis, keepdims=True)
    return tt.log(tt.sum(tt.exp(x - x_max), axis=axis, keepdims=True)) + x_max

class LinearNet(object):
    def __init__(self, n_inputs, n_outputs, n_components=1, 
                 seed=None, svi=True, diag_cov=False):
        """Initialize a linear-affine density model with homoscedastic noise

        Assumes that the input data (summary statistics) are centered on obs_stats,
        i.e. it fits only differences x - xo

        Parameters
        ----------
        n_inputs : int or tuple of ints or list of ints
            Dimensionality of input
        n_outputs : int
            Dimensionality of output
        n_components : int
            Number of components of the mixture density
        n_filters : list of ints
            Number of filters  per convolutional layer
        n_hiddens : list of ints
            Number of hidden units per fully connected layer
        n_rnn : None or int
            Number of RNN units
        impute_missing : bool
            If set to True, learns replacement value for NaNs, otherwise those
            inputs are set to zero
        seed : int or None
            If provided, random number generator will be seeded
        svi : bool
            Whether to use SVI version or not
        """

        assert n_components == 1
        self.n_components = n_components

        self.diag_cov = diag_cov
        self.n_outputs = n_outputs

        self.svi = svi # not used atm

        self.iws = tt.vector('iws', dtype=dtype)

        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

        self.params_dict_ = {

        'means.mW0' : np.zeros((n_inputs, n_outputs)),
        'means.mb0' : np.zeros((n_outputs)),

        'precisions.mW0' : np.zeros((n_inputs, n_outputs**2)),
        'precisions.mb0' : np.zeros((n_outputs))

        }


    def fit(self, trn_data, obs_stats):
        """ Initializes network with zero hidden layers.

        Without hidden layers, posterior means are linear functions Ax+b,
        and posterior precisions are exp(Cx + d)**2.

        We can initialize A,b,C,d from a homoscedastic linear fit assuming
        theta = f(x) = Ax + b + eps, where eps ~ N(0, Sig)
        and Sig = exp(d)**2, C = 0.
        We assume diagonal noise covariance Sig. 

        """
        assert self.n_components == 1
        assert self.diag_cov

        th, x, w = trn_data
        w = w.reshape(-1, 1)
        wth =  w * th

        # solve means
        X = np.hstack((np.ones((th.shape[0], 1)), x))
        beta = np.linalg.solve( X.T.dot(w * X), X.T.dot(wth))
        self.params_dict_['means.mW0'], self.params_dict_['means.mb0'] = beta[1:,:], beta[0,:]

        # solve variances
        Sig = (th.T.dot(wth) - X.dot(beta).T.dot(wth))/th.shape[0]

        self.params_dict_['precisions.mW0'] = np.zeros_like(self.params_dict['precisions.mW0'])
        self.params_dict_['precisions.mb0'] = - np.diag(np.log(np.sqrt(np.diag(Sig)))).reshape(-1)


    def get_mog(self, stats, deterministic=True):
        """Return the conditional MoG at location x

        Parameters
        ----------
        stats : np.array
            single input location
        deterministic : bool
            if True, mean weights are used for Bayesian network
        """
        assert stats.shape[0] == 1, 'x.shape[0] needs to be 1'

        a = np.ones(1)
        ms = [ self.get_mean(stats) ]
        Us = [ get_cov(self, stats) ]

        return dd.MoG(a=a, ms=ms, Us=Us, seed=self.gen_newseed())

    def get_mean(self, stats):
        return stats.dot(self.params_dict['means.mW0']) + np.atleast_2d(self.params_dict['means.mb0'])

    def get_cov(self, stats):
        if self.diag_cov:
            U = self.params_dict['precisions.mb0']
            return np.diag(np.exp(np.diag(U.reshape(self.n_outputs, self.n_outputs))))
        else:
            raise NotImplementedError

    def gen_newseed(self):
        """Generates a new random seed"""
        if self.seed is None:
            return None
        else:
            return self.rng.randint(0, 2**31)

    @property
    def params_dict(self):
        """Getter for params as dict"""
        pdict = {}
        for p in self.params_dict_:
            pdict[str(p)] = self.params_dict_[str(p)].copy()
        return pdict

    @params_dict.setter
    def params_dict(self, pdict):
        """Setter for params as dict"""
        for p in self.params_dict_:
            if str(p) in pdict.keys():
                self.params_dict_[str(p)] = pdict[str(p)].copy()

    @property
    def spec_dict(self):
        """Specs as dict"""
        return {'n_inputs': self.n_inputs,
                'n_outputs': self.n_outputs,
                'n_components': self.n_components,
                'seed': self.seed,
                'svi': self.svi}

    def get_loss(self):
        return np.nan
