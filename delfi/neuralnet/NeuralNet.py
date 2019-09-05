import collections
import delfi.distribution as dd
import delfi.neuralnet.layers as dl
import lasagne
import lasagne.layers as ll
import lasagne.nonlinearities as lnl
import numpy as np
import theano
import theano.tensor as tt
from copy import deepcopy
from snl.ml.models.mafs import ConditionalMaskedAutoregressiveFlow

from delfi.utils.data import isint
from delfi.utils.odict import first, last, nth
from delfi.utils.symbolic import MyLogSumExp, tensorN


dtype = theano.config.floatX


class NeuralNet(object):
    def __init__(self, n_inputs=None, n_outputs=None, input_shape=None,
                 n_bypass=0,
                 density='mog',
                 n_hiddens=(10, 10), impute_missing=True, seed=None,
                 n_filters=(), filter_sizes=3, pool_sizes=2,
                 n_rnn=0,
                 **density_opts):

        """Initialize a mixture density network with custom layers

        Parameters
        ----------
        n_inputs : int
            Total input dimensionality (data/summary stats)
        n_outputs : int
            Dimensionality of output (simulator parameters)
        input_shape : tuple
            Size to which data are reshaped before CNN or RNN
        n_bypass : int
            Number of elements at end of input which bypass CNN or RNN
        density : string
            Type of density condition on the network, can be 'mog' or 'maf'
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
        density_opts : dict
            Options for the density estimator
        """
        if n_rnn > 0 and len(n_filters) > 0:
            raise NotImplementedError
        assert isint(n_inputs) and isint(n_outputs)\
            and n_inputs > 0 and n_outputs > 0

        self.density = density.lower()
        self.impute_missing = impute_missing
        self.n_hiddens = list(n_hiddens)
        self.n_outputs, self.n_inputs = n_outputs, n_inputs
        self.n_bypass = n_bypass

        self.n_rnn = n_rnn

        self.n_filters, self.filter_sizes, self.pool_sizes, n_cnn = \
            list(n_filters), filter_sizes, pool_sizes, len(n_filters)
        if type(self.filter_sizes) is int:
            self.filter_sizes = [self.filter_sizes for _ in range(n_cnn)]
        else:
            assert len(self.filter_sizes) >= n_cnn
        if type(self.pool_sizes) is int:
            self.pool_sizes = [self.pool_sizes for _ in range(n_cnn)]
        else:
            assert len(self.pool_sizes) >= n_cnn

        self.iws = tt.vector('iws', dtype=dtype)

        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()
        lasagne.random.set_rng(self.rng)

        self.input_shape = (n_inputs,) if input_shape is None else input_shape
        assert np.prod(self.input_shape) + self.n_bypass == self.n_inputs
        assert 1 <= len(self.input_shape) <= 3

        # params: output placeholder (batch, self.n_outputs)
        self.params = tensorN(2, name='params', dtype=dtype)

        # stats : input placeholder, (batch, self.n_inputs)
        self.stats = tensorN(2, name='stats', dtype=dtype)

        # compose layers
        self.layer = collections.OrderedDict()

        # input layer, None indicates batch size not fixed at compile time
        self.layer['input'] = ll.InputLayer(
            (None, self.n_inputs), input_var=self.stats)

        # learn replacement values
        if self.impute_missing:
            self.layer['missing'] = \
                dl.ImputeMissingLayer(last(self.layer),
                                      n_inputs=(self.n_inputs,))
        else:
            self.layer['missing'] = \
                dl.ReplaceMissingLayer(last(self.layer),
                                       n_inputs=(self.n_inputs,))

        if self.n_bypass > 0 and (self.n_rnn > 0 or n_cnn > 0):
            last_layer = last(self.layer)
            bypass_slice = slice(self.n_inputs - self.n_bypass, self.n_inputs)
            direct_slice = slice(0, self.n_inputs - self.n_bypass)
            self.layer['bypass'] = ll.SliceLayer(last_layer, bypass_slice)
            self.layer['direct'] = ll.SliceLayer(last_layer, direct_slice)

        # reshape inputs prior to RNN or CNN step
        if self.n_rnn > 0 or n_cnn > 0:

            if len(n_filters) > 0 and len(self.input_shape) == 2:  # 1 channel
                rs = (-1, 1, *self.input_shape)
            else:
                if self.n_rnn > 0:
                    assert len(self.input_shape) == 2  # time, dim
                else:
                    assert len(self.input_shape) == 3  # channel, row, col
                rs = (-1, *self.input_shape)

            # last layer is 'missing' or 'direct'
            self.layer['reshape'] = ll.ReshapeLayer(last(self.layer), rs)

        # recurrent neural net, input: (batch, sequence_length, num_inputs)
        if self.n_rnn > 0:
            self.layer['rnn'] = ll.GRULayer(last(self.layer), n_rnn,
                                            only_return_final=True)

        # convolutional net, input: (batch, channels, rows, columns)
        if n_cnn > 0:
            for l in range(n_cnn):  # add layers
                if self.pool_sizes[l] == 1:
                    padding = (self.filter_sizes[l] - 1) // 2
                else:
                    padding = 0
                self.layer['conv_' + str(l + 1)] = ll.Conv2DLayer(
                    name='c' + str(l + 1),
                    incoming=last(self.layer),
                    num_filters=self.n_filters[l],
                    filter_size=self.filter_sizes[l],
                    stride=(1, 1),
                    pad=padding,
                    untie_biases=False,
                    W=lasagne.init.GlorotUniform(),
                    b=lasagne.init.Constant(0.),
                    nonlinearity=lnl.rectify,
                    flip_filters=True,
                    convolution=tt.nnet.conv2d)

                if self.pool_sizes[l] > 1:
                    self.layer['pool_' + str(l + 1)] = ll.MaxPool2DLayer(
                        name='p' + str(l + 1),
                        incoming=last(self.layer),
                        pool_size=self.pool_sizes[l],
                        stride=None,
                        ignore_border=True)

        # flatten
        self.layer['flatten'] = ll.FlattenLayer(
            incoming=last(self.layer),
            outdim=2)

        # incorporate bypass inputs
        if self.n_bypass > 0 and (self.n_rnn > 0 or n_cnn > 0):
            self.layer['bypass_merge'] = lasagne.layers.ConcatLayer(
                [self.layer['bypass'], last(self.layer)], axis=1)

        if self.density == 'mog':
            self.init_mdn(**density_opts)
        elif self.density == 'maf':
            self.init_maf(**density_opts)
        else:
            raise NotImplementedError

        self.compile_funs()  # theano functions

    def init_maf(self, n_mades=5, batch_norm=False, maf_actfun='tanh',
                 output_order='random', maf_mode='random',
                 **unused_kwargs):
        """
        :param n_mades:
        :param batch_norm:
        :param output_order:
        :param maf_mode:
        :param unused_kwargs:
        :return:
        """
        if batch_norm:
            raise NotImplementedError  # why?
        self.n_mades, self.batch_norm, self.output_order, self.maf_mode = \
            n_mades, batch_norm, output_order, maf_mode
        self.maf_actfun = maf_actfun
        for key in unused_kwargs.keys():
            print("CMAF ignoring unused input {0}".format(key))

        # get previous output/params
        self.maf_input = ll.get_output(last(self.layer))
        prev_params = ll.get_all_params(last(self.layer))
        input_shape_cmaf = last(self.layer).output_shape
        assert len(input_shape_cmaf) == 2  # (batch, input_dim)
        n_inputs_cmaf = input_shape_cmaf[1]

        rng_maf = np.random.RandomState(seed=self.gen_newseed())

        self.cmaf = ConditionalMaskedAutoregressiveFlow(
            n_inputs=n_inputs_cmaf,  n_outputs=self.n_outputs,
            n_hiddens=self.n_hiddens, act_fun=self.maf_actfun,
            n_mades=self.n_mades, batch_norm=self.batch_norm,
            output_order=self.output_order, mode=self.maf_mode,
            input=self.maf_input, output=self.params, rng=rng_maf)

        self.aps = prev_params + self.cmaf.parms
        self.lprobs = self.cmaf.L  # model log-likelihood
        self.dlprobs = self.lprobs  # svi not possible

    def init_mdn(self, svi=False, n_components=1, rank=None,
                 mdn_actfun=lnl.tanh, homoscedastic=False, min_precisions=None,
                 **unused_kwargs):
        """
        :param svi: bool
            Whether to use SVI version or not
        :param n_components: int
        :param rank: int
        :param homoscedastic: bool
        :param unused_kwargs: dict
        :param mdn_actfun: lasagne nonlinearity
            activation function for hidden units
        :param min_precisions: minimum values for diagonal elements of precision
            matrix for all components (usually taken to be prior precisions)
        :return: None
        """
        self.svi, self.n_components, self.rank, self.mdn_actfun,\
            self.homoscedastic, self.min_precisions = \
            svi, n_components, rank, mdn_actfun, homoscedastic, min_precisions
        for key in unused_kwargs.keys():
            print("MDN ignoring unused input {0}".format(key))

        # hidden layers
        for l in range(len(self.n_hiddens)):
            self.layer['hidden_' + str(l + 1)] = dl.FullyConnectedLayer(
                last(self.layer), n_units=self.n_hiddens[l],
                actfun=self.mdn_actfun,
                svi=self.svi, name='h' + str(l + 1))

        last_hidden = last(self.layer)
        # mixture layers
        self.layer['mixture_weights'] = dl.MixtureWeightsLayer(last_hidden,
            n_units=self.n_components, actfun=lnl.softmax, svi=self.svi,
            name='weights')
        self.layer['mixture_means'] = dl.MixtureMeansLayer(last_hidden,
            n_components=self.n_components, n_dim=self.n_outputs, svi=self.svi,
            name='means')
        if self.homoscedastic:
            PrecisionsLayer = dl.MixtureHomoscedasticPrecisionsLayer
        else:
            PrecisionsLayer = dl.MixturePrecisionsLayer
        # why is homoscedastic an input to the layer init?
        self.layer['mixture_precisions'] = PrecisionsLayer(last_hidden,
            n_components=self.n_components, n_dim=self.n_outputs, svi=self.svi,
            name='precisions', rank=self.rank, homoscedastic=self.homoscedastic,
            min_precisions=min_precisions)

        last_mog = [self.layer['mixture_weights'],
                    self.layer['mixture_means'],
                    self.layer['mixture_precisions']]

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
        self.lprobs = (MyLogSumExp(tt.stack(self.lprobs_comps, axis=1) + tt.log(self.a), axis=1)
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

        # weight and bias parameter sets as separate lists
        self.mps_wp = ll.get_all_params(last_mog, mp=True, wp=True)
        self.sps_wp = ll.get_all_params(last_mog, sp=True, wp=True)
        self.mps_bp = ll.get_all_params(last_mog, mp=True, bp=True)
        self.sps_bp = ll.get_all_params(last_mog, sp=True, bp=True)

    def compile_funs(self):
        """Compiles theano functions"""
        self._f_eval_lprobs = theano.function(
            inputs=[self.params, self.stats],
            outputs=self.lprobs)
        self._f_eval_dlprobs = theano.function(
            inputs=[self.params, self.stats],
            outputs=self.dlprobs)
        if self.density == 'mog':
            self._f_eval_comps = theano.function(
                inputs=[self.stats],
                outputs=self.comps)
            self._f_eval_dcomps = theano.function(
                inputs=[self.stats],
                outputs=self.dcomps)
        elif self.density == 'maf':
            self._f_eval_maf_input = theano.function(
                inputs=[self.stats],
                outputs=self.maf_input)

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
        assert self.density == 'mog'
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

    def get_density(self, stats, deterministic=True):
        assert stats.size == self.n_inputs
        stats = stats.reshape(1, -1).astype(dtype)
        if self.density == 'mog':
            return self.get_mog(stats, deterministic=deterministic)

        elif self.density == 'maf':
            assert deterministic
            cmaf_input = self._f_eval_maf_input(stats)
            return MAFconditional(
                model=self.cmaf, cmaf_inputs=cmaf_input.reshape(-1),
                makecopy=True,
                rng=np.random.RandomState(seed=self.gen_newseed()))

        else:
            raise NotImplementedError

    def get_mog(self, stats, deterministic=True):
        """Return the conditional MoG at location x

        Parameters
        ----------
        stats : np.array
            single input location
        deterministic : bool
            if True, mean weights are used for Bayesian network
        """
        assert self.density == 'mog'
        assert stats.shape[0] == 1, 'x.shape[0] needs to be 1'

        comps = self.eval_comps(stats, deterministic)
        a = comps['a'][0]
        ms = [comps['m' + str(i)][0] for i in range(self.n_components)]
        Us = [comps['U' + str(i)][0] for i in range(self.n_components)]

        return dd.MoG(a=a, ms=ms, Us=Us, seed=self.gen_newseed())

    def get_mog_tensors(self, svi=True, return_extras=False):
        assert self.density == 'mog'
        if svi:
            a, ms, Us, ldetUs = self.a, self.ms, self.Us, self.ldetUs
        else:
            a, ms, Us, ldetUs = self.da, self.dms, self.dUs, self.dldetUs
        if not return_extras:
            return a, ms, Us, ldetUs

        # precisions of posterior components:
        Ps = [tt.batched_dot(U.dimshuffle(0, 2, 1), U) for U in Us]
        # log determinants of posterior component precisions
        ldetPs = [2 * ldetU for ldetU in ldetUs]
        # precision times mean for each posterior component:
        Pms = [tt.sum(P * m.dimshuffle(0, 'x', 1), axis=2)
               for m, P in zip(ms, Ps)]
        # calculate tensorQF(P, m):
        QFs = [tt.sum(m * Pm, axis=1) for m, Pm in zip(ms, Pms)]
        return a, ms, Us, ldetUs, Ps, ldetPs, Pms, QFs

    def reseed(self, seed):
        """Reseeds the network's RNG"""
        self.rng.seed(seed=seed)
        self.seed = seed

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
        spec = dict(n_inputs=self.n_inputs, n_outputs=self.n_outputs,
                    n_filters=self.n_filters, n_hiddens=self.n_hiddens,
                    n_rnn=self.n_rnn, n_bypass=self.n_bypass,
                    input_shape=self.input_shape, filter_sizes=self.filter_sizes,
                    pool_sizes=self.pool_sizes, seed=self.seed)

        if self.density == 'mog':
            spec.update(dict(svi=self.svi, n_components=self.n_components,
                             rank=self.rank, homoscedastic=self.homoscedastic))
        elif self.density == 'maf':
            spec.update(dict(n_mades=5, batch_norm=False,
                             output_order='sequential', maf_mode='random'))
        else:
            raise NotImplementedError

        return spec

    def get_loss(self):
        return -tt.mean(self.iws * self.lprobs)


class MAFconditional(object):
    """Distribution-like object for masked autoregressive flows.

    Allows to sample and evaluate densities for parameters from the posterior
    estimate as required for SNPE(-C).
    """
    def __init__(self, model, cmaf_inputs, makecopy=False, rng=np.random):
        assert isinstance(model, ConditionalMaskedAutoregressiveFlow)
        self.model = deepcopy(model) if makecopy else model
        assert cmaf_inputs.ndim == 1
        self.cmaf_inputs = cmaf_inputs
        self.rng = rng
        # store a scale and offset so we can transform/normalize
        self.scale = np.ones(self.model.n_outputs)
        self.offset = np.zeros(self.model.n_outputs)
        self.ndim = self.model.n_outputs

    def set_scale_and_offset(self, scale=None, offset=None):
        if scale is not None:
            assert scale.ndim == 1 and scale.size == self.model.n_outputs and \
                np.isfinite(scale).all() and not np.any(scale == 0)
        if offset is not None:
            assert offset.ndim == 1 and offset.size == self.model.n_outputs and \
                np.isfinite(offset).all()
        if scale is not None:
            self.scale = scale
        if offset is not None:
            self.offset = offset

    def get_scale_and_offset(self):
        return self.scale, self.offset

    def gen(self, n_samples, cmaf_inputs=None):
        """Method to generate samples

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        cmaf_inputs : array
            Vector of inputs into cmaf (for conditional density param|stats )

        Returns
        -------
        n_samples x self.ndim
        """
        if cmaf_inputs is not None:
            assert cmaf_inputs.ndim == 1
        theta = self.model.gen(
            x=self.cmaf_inputs if cmaf_inputs is None else cmaf_inputs,
            n_samples=n_samples,
            rng=self.rng)
        return theta * self.scale + self.offset

    def eval(self, theta, log=True):
        """Method to evaluate pdf

        Parameters
        ----------
        theta : int or list or np.array
            Rows are inputs to evaluate at
        log : bool, defaulting to True
            If True, the log pdf is evaluated

        Returns
        -------
        scalar
        """
        assert theta.ndim < 3
        theta_norm = (theta - self.offset) / self.scale
        logp = self.model.eval(xy=(self.cmaf_inputs, theta_norm), log=True) \
            - np.sum(np.log(np.abs(self.scale)))
        return logp if log else np.exp(logp)
