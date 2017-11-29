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

class NeuralNet(object):
    def __init__(self, n_inputs, n_outputs, n_components=1, n_filters=[],
                 n_hiddens=[10, 10], n_rnn=None, impute_missing=True, seed=None,
                 svi=True):
        """Initialize a mixture density network with custom layers

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
        self.impute_missing = impute_missing
        self.n_components = n_components
        self.n_filters = n_filters
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.svi = svi

        if n_rnn is None:
            self.n_rnn = 0
        else:
            self.n_rnn = n_rnn
        if self.n_rnn > 0 and len(self.n_filters) > 0:
            raise NotImplementedError

        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()
        lasagne.random.set_rng(self.rng)

        # cast n_inputs to tuple
        if type(n_inputs) is int:
            self.n_inputs = (n_inputs, )
        elif type(n_inputs) is list:
            self.n_inputs = tuple(n_inputs)
        elif type(n_inputs) is tuple:
            self.n_inputs = n_inputs
        else:
            raise ValueError('n_inputs type not supported')

        # compose layers
        self.layer = collections.OrderedDict()

        # stats : input placeholder, (batch, *self.n_inputs)
        if len(self.n_inputs)+1 == 2:
            self.stats = tt.matrix('stats', dtype=dtype)
        elif len(self.n_inputs)+1 == 3:
            self.stats = tt.tensor3('stats', dtype=dtype)
        elif len(self.n_inputs)+1 == 4:
            self.stats = tt.tensor4('stats', dtype=dtype)
        else:
            raise NotImplementedError

        # input layer
        self.layer['input'] = ll.InputLayer(
            (None, *self.n_inputs), input_var=self.stats)

        # learn replacement values
        if self.impute_missing:
            self.layer['missing'] = dl.ImputeMissingLayer(last(self.layer),
                                                          n_inputs=self.n_inputs)
        else:
            self.layer['missing'] = dl.ReplaceMissingLayer(last(self.layer),
                                                           n_inputs=self.n_inputs)

        # recurrent neural net
        # expects shape (batch, sequence_length, num_inputs)
        if self.n_rnn > 0:
            if len(self.n_inputs) == 1:
                rs = (-1, *self.n_inputs, 1)
                self.layer['rnn_reshape'] = ll.ReshapeLayer(last(self.layer), rs)

            self.layer['rnn'] = ll.GRULayer(last(self.layer), n_rnn,
                                            only_return_final=True)

        # convolutional layers
        # expects shape (batch, num_input_channels, input_rows, input_columns)
        if len(self.n_filters) > 0:
            # reshape
            if len(self.n_inputs) == 1:
                raise NotImplementedError
            elif len(self.n_inputs) == 2:
                rs = (-1, 1, *self.n_inputs)
            else:
                rs = None
            if rs is not None:
                self.layer['conv_reshape'] = ll.ReshapeLayer(last(self.layer), rs)

            # add layers
            for l in range(len(n_filters)):
                self.layer['conv_' + str(l + 1)] = ll.Conv2DLayer(
                    name='c' + str(l + 1),
                    incoming=last(self.layer),
                    num_filters=n_filters[l],
                    filter_size=3,
                    stride=(2, 2),
                    pad=0,
                    untie_biases=False,
                    W=lasagne.init.GlorotUniform(),
                    b=lasagne.init.Constant(0.),
                    nonlinearity=lnl.rectify,
                    flip_filters=True,
                    convolution=tt.nnet.conv2d)

        # flatten
        self.layer['flatten'] = ll.FlattenLayer(
            incoming=last(self.layer),
            outdim=2)

        # hidden layers
        for l in range(len(n_hiddens)):
            self.layer['hidden_' + str(l + 1)] = dl.FullyConnectedLayer(
                last(self.layer), n_units=n_hiddens[l],
                svi=svi, name='h' + str(l + 1))

        last_hidden = last(self.layer)

        # mixture layers
        self.layer['mixture_weights'] = dl.MixtureWeightsLayer(
            last_hidden, n_units=n_components, actfun=lnl.softmax, svi=svi,
            name='weights')
        self.layer['mixture_means'] = dl.MixtureMeansLayer(
            last_hidden, n_components=n_components, n_dim=n_outputs, svi=svi,
            name='means')
        self.layer['mixture_precisions'] = dl.MixturePrecisionsLayer(
            last_hidden, n_components=n_components, n_dim=n_outputs, svi=svi,
            name='precisions')
        last_mog = [self.layer['mixture_weights'],
                    self.layer['mixture_means'],
                    self.layer['mixture_precisions']]

        # output placeholder
        self.params = tt.matrix('params', dtype=dtype)  # (batch, self.n_outputs)

        # mixture parameters
        # a : weights, matrix with shape (batch, n_components)
        # ms : means, list of len n_components with (batch, n_dim, n_dim)
        # Us : precision factors, n_components list with (batch, n_dim, n_dim)
        # ldetUs : log determinants of precisions, n_comp list with (batch, )
        self.a, self.ms, precision_out = ll.get_output(last_mog,
                                                       deterministic=False)

        self.Us = precision_out['Us']
        self.ldetUs = precision_out['ldetUs']

        self.comps = {
            **{'a': self.a},
            **{'m' + str(i): self.ms[i] for i in range(self.n_components)},
            **{'U' + str(i): self.Us[i] for i in range(self.n_components)}}

        # log probability of y given the mixture distribution
        # lprobs_comps : log probs per component, list of len n_components with (batch, )
        # probs : log probs of mixture, (batch, )

        self.lprobs_comps = [-0.5 * tt.sum(tt.sum((self.params - m).dimshuffle(
            [0, 'x', 1]) * U, axis=2)**2, axis=1) + ldetU
            for m, U, ldetU in zip(self.ms, self.Us, self.ldetUs)]
        self.lprobs = (MyLogSumExp(tt.stack(self.lprobs_comps, axis=1) + tt.log(self.a), axis=1) \
                      - (0.5 * self.n_outputs * np.log(2 * np.pi))).squeeze()

        # the quantities from above again, but with deterministic=True
        # --- in the svi case, this will disable injection of randomness;
        # the mean of weights is used instead
        self.da, self.dms, dprecision_out = ll.get_output(last_mog,
                                                          deterministic=True)
        self.dUs = dprecision_out['Us']
        self.dldetUs = dprecision_out['ldetUs']
        self.dcomps = {
            **{'a': self.da},
            **{'m' + str(i): self.dms[i] for i in range(self.n_components)},
            **{'U' + str(i): self.dUs[i] for i in range(self.n_components)}}
        self.dlprobs_comps = [-0.5 * tt.sum(tt.sum((self.params - m).dimshuffle(
            [0, 'x', 1]) * U, axis=2)**2, axis=1) + ldetU
            for m, U, ldetU in zip(self.dms, self.dUs, self.dldetUs)]
        self.dlprobs = (MyLogSumExp(tt.stack(self.dlprobs_comps, axis=1) + tt.log(self.da), axis=1) \
                       - (0.5 * self.n_outputs * np.log(2 * np.pi))).squeeze()

        # parameters of network
        self.aps = ll.get_all_params(last_mog)  # all parameters
        self.mps = ll.get_all_params(last_mog, mp=True)  # means
        self.sps = ll.get_all_params(last_mog, sp=True)  # log stds

        # weight and bias parameter sets as seperate lists
        self.mps_wp = ll.get_all_params(last_mog, mp=True, wp=True)
        self.sps_wp = ll.get_all_params(last_mog, sp=True, wp=True)
        self.mps_bp = ll.get_all_params(last_mog, mp=True, bp=True)
        self.sps_bp = ll.get_all_params(last_mog, sp=True, bp=True)

        # theano functions
        self.compile_funs()

    def compile_funs(self):
        """Compiles theano functions"""
        self._f_eval_comps = theano.function(
            inputs=[self.stats],
            outputs=self.comps)
        self._f_eval_lprobs = theano.function(
            inputs=[self.params, self.stats],
            outputs=self.lprobs)
        self._f_eval_dcomps = theano.function(
            inputs=[self.stats],
            outputs=self.dcomps)
        self._f_eval_dlprobs = theano.function(
            inputs=[self.params, self.stats],
            outputs=self.dlprobs)

    def eval_comps(self, stats, deterministic=True):
        """Evaluate the parameters of all mixture components at given inputs

        Parameters
        ----------
        stats : np.array
            rows are input locations
        deterministic : bool
            if True, mean weights are used for Bayesian network

        Returns
        -------
        mixing coefficients, means and scale matrices
        """
        if deterministic:
            return self._f_eval_dcomps(stats.astype(dtype))
        else:
            return self._f_eval_comps(stats.astype(dtype))

    def eval_lprobs(self, params, stats, deterministic=True):
        """Evaluate log probabilities for given input-output pairs.

        Parameters
        ----------
        params : np.array
        stats : np.array
        deterministic : bool
            if True, mean weights are used for Bayesian network

        Returns
        -------
        log probabilities : log p(params|stats)
        """
        if deterministic:
            return self._f_eval_dlprobs(params.astype(dtype), stats.astype(dtype))
        else:
            return self._f_eval_lprobs(params.astype(dtype), stats.astype(dtype))

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

        comps = self.eval_comps(stats, deterministic)
        a = comps['a'][0]
        ms = [comps['m' + str(i)][0] for i in range(self.n_components)]
        Us = [comps['U' + str(i)][0] for i in range(self.n_components)]

        return dd.MoG(a=a, ms=ms, Us=Us, seed=self.gen_newseed())

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
        for p in self.aps:
            pdict[str(p)] = p.get_value()
        return pdict

    @params_dict.setter
    def params_dict(self, pdict):
        """Setter for params as dict"""
        for p in self.aps:
            if str(p) in pdict.keys():
                p.set_value(pdict[str(p)])

    @property
    def spec_dict(self):
        """Specs as dict"""
        return {'n_inputs': self.n_inputs,
                'n_outputs': self.n_outputs,
                'n_components': self.n_components,
                'n_filters': self.n_filters,
                'n_hiddens': self.n_hiddens,
                'n_rnn': self.n_rnn,
                'seed': self.seed,
                'svi': self.svi}
