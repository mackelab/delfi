import collections
import delfi.distribution as dd
import delfi.neuralnet.layers as dl

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from delfi.neuralnet.layers.Layer import Layer, FlattenLayer
from delfi.utils.odict import first, last, nth

dtype = torch.DoubleTensor

def MyLogSumExp(x, axis):
    x_max = torch.max(x, dim=axis, keepdim=True)[0]
    return torch.log(torch.sum(torch.exp(x - x_max), dim=axis, keepdim=True)) + x_max

class NeuralNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_components=1, svi=False,
                 n_hiddens=[10, 10], seed=None):
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
        self.n_components = n_components
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs

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
        self.layer['missing'] = dl.ReplaceMissingLayer((None, *self.n_inputs))

        # flatten
        self.layer['flatten'] = FlattenLayer(
            incoming=last(self.layer),
            outdim=2)

        # hidden layers
        for l in range(len(n_hiddens)):
            self.layer['hidden_' + str(l + 1)] = dl.FullyConnectedLayer(
                last(self.layer), n_units=n_hiddens[l],
                name='h' + str(l + 1))

        self.last_hidden = last(self.layer)

        # mixture layers
        self.layer['mixture_weights'] = dl.MixtureWeightsLayer(
            self.last_hidden, n_units=n_components, actfun=torch.nn.functional.softmax,
            name='weights')
        self.layer['mixture_means'] = dl.MixtureMeansLayer(
            self.last_hidden, n_components=n_components, n_dim=n_outputs,
            name='means')
        self.layer['mixture_precisions'] = dl.MixturePrecisionsLayer(
            self.last_hidden, n_components=n_components, n_dim=n_outputs,
            name='precisions')

        for ln in self.layer:
            self.add_module(ln, self.layer[ln])

        self.svi=False
        self.lprobs = []
        self.params = []
        self.stats = []
        self.aps = {}
        print(list(self.parameters()))

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
        x = Variable(dtype(stats.flatten().astype('double')).view(*stats.shape))
        for l in self.layer:
            x = self.layer[l](x)
            if self.layer[l] == self.last_hidden:
                break

        a = self.layer['mixture_weights'](x)
        ms = self.layer['mixture_means'](x)
        prec_data = self.layer['mixture_precisions'](x)
        Us, ldetUs = prec_data['Us'], prec_data['ldetUs']
    
        return a, ms, Us, ldetUs

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
        params = Variable(dtype(params.flatten().astype('double')).view(*params.shape))
        a, ms, Us, ldetUs = self.eval_comps(stats)
        comps = {
            **{'a': a},
            **{'m' + str(i): ms[i] for i in range(self.n_components)},
            **{'U' + str(i): Us[i] for i in range(self.n_components)}}

        for k in comps:
            print("{} : {}".format(k, comps[k][0]))

        m = ms[0]
        U = Us[0]
        lprobs_comps = [-0.5 * torch.sum(torch.sum((params - m).unsqueeze
            (1) * U, dim=2)**2, dim=1) + ldetU
            for m, U, ldetU in zip(ms, Us, ldetUs)]

        lprobs = MyLogSumExp(torch.stack(lprobs_comps, dim=1) + torch.log(a) \
                       - (0.5 * self.n_outputs * np.log(2 * np.pi)), axis=1).squeeze()

        self.lprobs = lprobs
        self.stats = stats
        self.params= params
           
        print("lprobs: {}".format(lprobs))

        return lprobs

    def forward(self, inp):
        params = inp[0]
        stats = inp[1]
        iws = Variable(dtype(inp[2]))
        iws = Variable(torch.ones(iws.size()).type(dtype))
        lprobs = self.eval_lprobs(params, stats)
        print(iws)
        print(lprobs * iws)
        ret = torch.sum(lprobs * iws)
        print("RETURNED: {}".format(ret))
        return ret

    def get_mog(self, stats):
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
