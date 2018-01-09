import collections
import delfi.distribution as dd
import delfi.neuralnet.layers as dl

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pdb

from delfi.neuralnet.layers.Layer import Layer, FlattenLayer, ReshapeLayer
from delfi.utils.odict import first, last, nth

dtype = torch.DoubleTensor

def MyLogSumExp(x, axis):
    x_max = torch.max(x, dim=axis, keepdim=True)[0]
    return torch.log(torch.sum(torch.exp(x - x_max), dim=axis, keepdim=True)) + x_max

class NeuralNet(nn.Module):
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
        n_hiddens : list of ints
            Number of hidden units per fully connected layer
        seed : int or None
            If provided, random number generator will be seeded
        """
        super().__init__()
        self.impute_missing = impute_missing
        self.n_components = n_components
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs
        self.n_filters = n_filters
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

        # learn replacement values
        if self.impute_missing:
            self.layer['missing'] = dl.ImputeMissingLayer((None, *self.n_inputs), n_inputs=self.n_inputs)
        else:
            self.layer['missing'] = dl.ReplaceMissingLayer((None, *self.n_inputs))

        
        # recurrent neural net
        # expects shape (batch, sequence_length, num_inputs)
        if self.n_rnn > 0:
            if len(self.n_inputs) == 1:
                rs = (-1, *self.n_inputs, 1)
                self.layer['rnn_reshape'] = ReshapeLayer(last(self.layer), rs)

            raise NotImplementedError
#             self.layer['rnn'] = ll.GRULayer(last(self.layer), n_rnn,
#                                             only_return_final=True)

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
                self.layer['conv_reshape'] = ReshapeLayer(last(self.layer), rs)

            raise NotImplementedError
#             # add layers
#             for l in range(len(n_filters)):
#                 self.layer['conv_' + str(l + 1)] = Conv2DLayer(
#                     name='c' + str(l + 1),
#                     incoming=last(self.layer),
#                     num_filters=n_filters[l],
#                     filter_size=3,
#                     stride=(2, 2),
#                     pad=0,
#                     untie_biases=False,
#                     W=lasagne.init.GlorotUniform(),
#                     b=lasagne.init.Constant(0.),
#                     nonlinearity=lnl.rectify,
#                     flip_filters=True,
#                     convolution=tt.nnet.conv2d)

        self.layer['flatten'] = FlattenLayer(
            incoming=last(self.layer),
            outdim=2)

        # hidden layers
        for l in range(len(n_hiddens)):
            self.layer['hidden_' + str(l + 1)] = dl.FullyConnectedLayer(
                last(self.layer), n_units=n_hiddens[l],
                svi=svi, name='h' + str(l + 1), seed=seed)

        self.last_hidden = last(self.layer)

        # mixture layers
        self.layer['mixture_weights'] = dl.MixtureWeightsLayer(
            self.last_hidden, n_units=n_components, actfun=torch.nn.functional.softmax, svi=svi,
            name='weights')
        self.layer['mixture_means'] = dl.MixtureMeansLayer(
            self.last_hidden, n_components=n_components, n_dim=n_outputs, svi=svi,
            name='means')
        self.layer['mixture_precisions'] = dl.MixturePrecisionsLayer(
            self.last_hidden, n_components=n_components, n_dim=n_outputs, svi=svi,
            name='precisions')

        for ln in self.layer:
            self.add_module(ln, self.layer[ln])

        self.lprobs = Variable(dtype([]))
        self.params = Variable(dtype([]))
        self.stats = Variable(dtype([]))
        self.iws = Variable(dtype([]))
        self.aps = {}
        
        last_mog = [self.layer['mixture_weights'],
                    self.layer['mixture_means'],
                    self.layer['mixture_precisions']]

        self.aps = get_params(last_mog)
        self.mps = get_params(last_mog, mp=True) 
        self.sps = get_params(last_mog, sp=True)

        self.mps_wp = get_params(last_mog, mp=True, wp=True)
        self.sps_wp = get_params(last_mog, sp=True, wp=True)
        self.mps_bp = get_params(last_mog, mp=True, bp=True)
        self.sps_bp = get_params(last_mog, sp=True, bp=True)

    def eval_comps(self, stats):
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
        if type(stats) == np.ndarray:
            x = Variable(dtype(stats.flatten().astype(dtype)).view(*stats.shape))
        else:
            x = stats

        for l in self.layer:
            x = self.layer[l](x)
            if self.layer[l] == self.last_hidden:
                break

        a = self.layer['mixture_weights'](x)
        ms = self.layer['mixture_means'](x)
        prec_data = self.layer['mixture_precisions'](x)
        Us, ldetUs = prec_data['Us'], prec_data['ldetUs']
    
        if type(stats) == np.ndarray:
            a = a.data.numpy()
            ms = [ m.data.numpy() for m in ms ]
            Us = [ U.data.numpy() for U in Us ]
            ldetUs = [ ldetU.data.numpy() for ldetU in ldetUs ]

        ret = {
            **{'a': a},
            **{'m' + str(i): ms[i] for i in range(self.n_components)},
            **{'U' + str(i): Us[i] for i in range(self.n_components)},
            **{'ldetU' + str(i): ldetUs[i] for i in range(self.n_components)}}

        return ret

    def eval_lprobs(self, params, stats):
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
        comps = self.eval_comps(stats)

        a = comps['a']
        ms = [ comps['m{}'.format(i)] for i in range(self.n_components) ]
        Us = [ comps['U{}'.format(i)] for i in range(self.n_components) ]
        ldetUs = [ comps['ldetU{}'.format(i)] for i in range(self.n_components) ]

        lprobs_comps = [-0.5 * torch.sum(torch.sum((params - m).unsqueeze
            (1) * U, dim=2)**2, dim=1) + ldetU
            for m, U, ldetU in zip(ms, Us, ldetUs)]

        lprobs = MyLogSumExp(torch.stack(lprobs_comps, dim=1) + torch.log(a) \
                       - (0.5 * self.n_outputs * np.log(2 * np.pi)), axis=1).squeeze()

        return lprobs

    def get_loss(self):
        if len(self.iws.size()) == 0:
            return Variable(dtype([0]))

        return -torch.mm(self.iws, self.lprobs)

    def forward(self, inp):
        params = Variable(dtype(inp[0].astype('double')))
        stats = Variable(dtype(inp[1].astype('double')))
        iws = Variable(dtype(inp[2].astype('double')))
        lprobs = self.eval_lprobs(params, stats)
        ret = torch.dot(lprobs, iws)

        self.lprobs = lprobs
        self.stats = stats
        self.params= params
        self.iws = iws

        return ret

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

        comps = self.eval_comps(stats)
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

def get_params(layers, **kwargs):
    return [ x for l in layers for x in l.get_params(**kwargs) ]
