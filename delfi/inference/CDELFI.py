import delfi.distribution as dd
import numpy as np
import theano.tensor as tt

from delfi.inference.BaseInference import BaseInference
from delfi.neuralnet.NeuralNet import NeuralNet
from delfi.neuralnet.Trainer import Trainer
from delfi.neuralnet.loss.regularizer import svi_kl_zero


class CDELFI(BaseInference):
    def __init__(self, generator, obs, prior_norm=False, pilot_samples=100,
                 n_components=1, reg_lambda=0.01, seed=None, verbose=True,
                 **kwargs):
        """Conditional density estimation likelihood-free inference (CDE-LFI)

        Implementation of algorithms 1 and 2 of Papamakarios and Murray, 2016.

        Parameters
        ----------
        generator : generator instance
            Generator instance
        obs : array
            Observation in the format the generator returns (1 x n_summary)
        prior_norm : bool
            If set to True, will z-transform params based on mean/std of prior
        pilot_samples : None or int
            If an integer is provided, a pilot run with the given number of
            samples is run. The mean and std of the summary statistics of the
            pilot samples will be subsequently used to z-transform summary
            statistics.
        n_components : int
            Number of components in final round (PM's algorithm 2)
        reg_lambda : float
            Precision parameter for weight regularizer if svi is True
        seed : int or None
            If provided, random number generator will be seeded
        verbose : bool
            Controls whether or not progressbars are shown
        kwargs : additional keyword arguments
            Additional arguments for the NeuralNet instance, including:
                n_hiddens : list of ints
                    Number of hidden units per layer of the neural network
                svi : bool
                    Whether to use SVI version of the network or not

        Attributes
        ----------
        observables : dict
            Dictionary containing theano variables that can be monitored while
            training the neural network.
        """
        # Algorithm 1 of PM requires a single component
        kwargs.update({'n_components': 1})

        super().__init__(generator, prior_norm=prior_norm,
                         pilot_samples=pilot_samples, seed=seed,
                         verbose=verbose, **kwargs)

        self.n_components = n_components
        self.obs = obs
        self.reg_lambda = reg_lambda

    def loss(self, N):
        """Loss function for training

        Parameters
        ----------
        N : int
            Number of training samples
        """
        loss = -tt.mean(self.network.lprobs)

        if self.svi:
            kl, imvs = svi_kl_zero(self.network.mps, self.network.sps,
                                   self.reg_lambda)
            loss = loss + 1 / N * kl

            # adding nodes to dict s.t. they can be monitored
            self.observables['loss.kl'] = kl
            self.observables.update(imvs)

        return loss

    def run(self, n_train=100, n_rounds=2, epochs=100, minibatch=50,
            monitor=None, **kwargs):
        """Run algorithm

        Parameters
        ----------
        n_train : int or list of ints
            Number of data points drawn per round. If a list is passed, the
            nth list element specifies the number of training examples in the
            nth round. If there are fewer list elements than rounds, the last
            list element is used.
        n_rounds : int
            Number of rounds
        epochs: int
            Number of epochs used for neural network training
        minibatch: int
            Size of the minibatches used for neural network training
        monitor : list of str
            Names of variables to record during training along with the value
            of the loss function. The observables attribute contains all
            possible variables that can be monitored
        kwargs : additional keyword arguments
            Additional arguments for the Trainer instance

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged while training the networks
        trn_datasets : list of (params, stats)
            training datasets, z-transformed
        posteriors : list of posteriors
            posterior after each round
        """
        logs = []
        trn_datasets = []
        posteriors = []

        for r in range(1, n_rounds + 1):  # start at 1
            # if round > 1, set new proposal distribution before sampling
            if r > 1:
                # posterior becomes new proposal prior
                posterior = self.predict(self.obs)
                self.generator.proposal = posterior.project_to_gaussian()

            # number of training examples for this round
            if type(n_train) == list:
                try:
                    n_train_round = n_train[r-1]
                except:
                    n_train_round = n_train[-1]
            else:
                n_train_round = n_train

            # draw training data (z-transformed params and stats)
            verbose = '(round {}) '.format(r) if self.verbose else False
            trn_data = self.gen(n_train_round, verbose=verbose)

            # algorithm 2 of Papamakarios and Murray
            if r == n_rounds and self.n_components > 1:
                # get parameters of current network
                old_params = self.network.params_dict.copy()

                # create new network
                network_spec = self.network.spec_dict.copy()
                network_spec.update({'n_components': self.n_components})
                self.network = NeuralNet(**network_spec)
                new_params = self.network.params_dict

                # set weights of new network
                # weights of additional components are duplicates
                for p in [s for s in new_params if 'means' in s or
                          'precisions' in s]:
                    new_params[p] = old_params[p[:-1] + '0']
                    new_params[p] += 1.0e-6*self.rng.randn(*new_params[p].shape)

                self.network.params_dict = new_params

            trn_inputs = [self.network.params, self.network.stats]

            t = Trainer(self.network, self.loss(N=n_train_round),
                        trn_data=trn_data, trn_inputs=trn_inputs,
                        monitor=self.monitor_dict_from_names(monitor),
                        seed=self.gen_newseed(), **kwargs)
            logs.append(t.train(epochs=epochs, minibatch=minibatch,
                                verbose=verbose))
            trn_datasets.append(trn_data)

            posteriors.append(self.predict(self.obs))

        return logs, trn_datasets, posteriors

    def predict(self, x):
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        """
        if self.generator.proposal is None:
            # no correction necessary
            return super(CDELFI, self).predict(x)  # via super
        else:
            # mog is posterior given proposal prior
            mog = super(CDELFI, self).predict(x)  # via super

            # compute posterior given prior by analytical division step
            if 'Uniform' in str(type(self.generator.prior)):
                posterior = mog / self.generator.proposal
            elif 'Gaussian' in str(type(self.generator.prior)):
                posterior = (mog * self.generator.prior) / \
                    self.generator.proposal
            else:
                raise NotImplemented

            return posterior
