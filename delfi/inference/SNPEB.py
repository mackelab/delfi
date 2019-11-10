import numpy as np

from delfi.inference.BaseInference import BaseInference
from delfi.neuralnet.Trainer import Trainer
from delfi.neuralnet.loss.regularizer import svi_kl_init, svi_kl_zero


class SNPEB(BaseInference):
    def __init__(self, generator, obs, prior_norm=False, pilot_samples=100,
                 convert_to_T=3, reg_lambda=0.01, prior_mixin=0, kernel=None, seed=None, verbose=True,
                 **kwargs):
        """SNPE-B

        Implementation of Lueckmann, Goncalves, Bassetto, Öcal, Nonnenmacher & Macke (NeurIPS 2017)

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
        convert_to_T : None or int
            Convert proposal distribution to Student's T? If a number if given,
            the number specifies the degrees of freedom. None for no conversion
        reg_lambda : float
            Precision parameter for weight regularizer if svi is True
        prior_mixin : float
            Percentage of the prior mixed into the proposal prior. While training,
            an additional prior_mixin * N samples will be drawn from the actual prior
            in each round.
        seed : int or None
            If provided, random number generator will be seeded
        verbose : bool
            Controls whether or not progressbars are shown
        kwargs : additional keyword arguments
            Additional arguments for the NeuralNet instance, including:
                n_components : int
                    Number of components of the mixture density
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
        assert obs is not None, "SNPE requires observed data"
        self.obs = np.asarray(obs)
        super().__init__(generator, prior_norm=prior_norm,
                         pilot_samples=pilot_samples, seed=seed,
                         verbose=verbose, **kwargs)

        if np.any(np.isnan(self.obs)):
            raise ValueError("Observed data contains NaNs")

        self.reg_lambda = reg_lambda
        self.convert_to_T = convert_to_T

        self.prior_mixin = 0 if prior_mixin is None else prior_mixin

        self.kernel = kernel

    def loss(self, N, round_cl=1):
        """Loss function for training

        Parameters
        ----------
        N : int
            Number of training samples
        """
        loss = self.network.get_loss()

        # adding nodes to dict s.t. they can be monitored during training
        self.observables['loss.lprobs'] = self.network.lprobs
        self.observables['loss.iws'] = self.network.iws
        self.observables['loss.raw_loss'] = loss

        if self.svi:
            if self.round <= round_cl:
                # weights close to zero-centered prior in the first round
                if self.reg_lambda > 0:
                    kl, imvs = svi_kl_zero(self.network.mps, self.network.sps,
                                           self.reg_lambda)
                else:
                    kl, imvs = 0, {}
            else:
                # weights close to those of previous round
                kl, imvs = svi_kl_init(self.network.mps, self.network.sps)

            loss = loss + 1 / N * kl

            # adding nodes to dict s.t. they can be monitored
            self.observables['loss.kl'] = kl
            self.observables.update(imvs)

        return loss

    def run(self, n_train=100, n_rounds=2, epochs=100, minibatch=50,
            round_cl=1, stop_on_nan=False, proposal=None,
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
        epochs : int
            Number of epochs used for neural network training
        minibatch : int
            Size of the minibatches used for neural network training
        monitor : list of str
            Names of variables to record during training along with the value
            of the loss function. The observables attribute contains all
            possible variables that can be monitored
        round_cl : int
            Round after which to start continual learning
        stop_on_nan : bool
            If True, will halt if NaNs in the loss are encountered
        proposal : Distribution of None
            If given, will use this distribution as the starting proposal prior
        kwargs : additional keyword arguments
            Additional arguments for the Trainer instance

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged while training the networks
        trn_datasets : list of (params, stats)
            training datasets, z-transformed
        posteriors : list of distributions
            posterior after each round
        """
        logs = []
        trn_datasets = []
        posteriors = []

        for r in range(n_rounds):
            self.round += 1

            if r == 0 and proposal is not None:
                self.generator.proposal = proposal
            # if round > 1, set new proposal distribution before sampling
            elif self.round > 1:
                # posterior becomes new proposal prior
                proposal = self.predict(self.obs)  # see super

                # convert proposal to student's T?
                if self.convert_to_T is not None:
                    if type(self.convert_to_T) == int:
                        dofs = self.convert_to_T
                    else:
                        dofs = 10
                    proposal = proposal.convert_to_T(dofs=dofs)

                self.generator.proposal = proposal

            # number of training examples for this round
            if type(n_train) == list:
                try:
                    n_train_round = n_train[self.round-1]
                except:
                    n_train_round = n_train[-1]
            else:
                n_train_round = n_train


            # draw training data (z-transformed params and stats)
            verbose = '(round {}) '.format(self.round) if self.verbose else False

            trn_data = self.gen(n_train_round, prior_mixin=self.prior_mixin, verbose=verbose)
            n_train_round = trn_data[0].shape[0]

            # precompute importance weights
            if self.generator.proposal is not None:
                params = self.params_std * trn_data[0] + self.params_mean
                p_prior = self.generator.prior.eval(params, log=False)
                p_proposal = self.generator.proposal.eval(params, log=False)
                iws = p_prior / (self.prior_mixin * p_prior + (1 - self.prior_mixin) * p_proposal)
            else:
                iws = np.ones((n_train_round,))

            # normalize weights
            iws /= np.mean(iws)

            if self.kernel is not None:
                iws *= self.kernel.eval(trn_data[1].reshape(n_train_round, -1))

            trn_data = (trn_data[0], trn_data[1], iws)
            trn_inputs = [self.network.params, self.network.stats,
                          self.network.iws]

            t = Trainer(self.network,
                        self.loss(N=n_train_round, round_cl=round_cl),
                        trn_data=trn_data, trn_inputs=trn_inputs,
                        seed=self.gen_newseed(),
                        monitor=self.monitor_dict_from_names(monitor),
                        **kwargs)
            logs.append(t.train(epochs=epochs, minibatch=minibatch,
                                verbose=verbose, stop_on_nan=stop_on_nan))

            trn_datasets.append(trn_data)

            try:
                posteriors.append(self.predict(self.obs))
            except np.linalg.LinAlgError:
                posteriors.append(None)
                print("Cannot predict posterior after round {} due to NaNs".format(r))
                break

        return logs, trn_datasets, posteriors
