import collections
import delfi.distribution as dd
import delfi.neuralnet.layers as dl

import torch
import torch.nn as nn

from delfi.neuralnet.Layer import Layer

dtype = torch.DoubleTensor

def MyLogSumExp(x, axis=None):
    x_max = torch.max(x, dim=axis, keepdims=True)
    return torch.log(torch.sum(torch.exp(x - x_max), dim=axis, keepdims=True)) + x_max

class NeuralNet(object):
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
            last_hidden, n_units=n_components, actfun=torch.softmax,
            name='weights')
        self.layer['mixture_means'] = dl.MixtureMeansLayer(
            last_hidden, n_components=n_components, n_dim=n_outputs,
            name='means')
        self.layer['mixture_precisions'] = dl.MixturePrecisionsLayer(
            last_hidden, n_components=n_components, n_dim=n_outputs,
            name='precisions')

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
        x = stats
        for l in self.layers:
            x = l(x)
            if l == self.last_hidden:
                break

        a = self.layer['mixture_weights'](x)
        ms = self.layer['mixture_means'](x)
        Us, ldetUs = self.layer['mixture_precisions'](x)
    
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
        a, ms, Us, ldetUs = self.eval_comps(stats)
        comps = {
            **{'a': a},
            **{'m' + str(i): ms[i] for i in range(self.n_components)},
            **{'U' + str(i): Us[i] for i in range(self.n_components)}}

        lprobs_comps = [-0.5 * torch.sum(torch.sum((self.params - m).unsqueeze
            (1) * U, dim=2)**2, dim=1) + ldetU
            for m, U, ldetU in zip(ms, Us, ldetUs)]

        lprobs = MyLogSumExp(torch.stack(lprobs_comps, dim=1) + torch.log(a, dim=1) \
                       - (0.5 * self.n_outputs * np.log(2 * np.pi))).squeeze()

        return lprobs

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
