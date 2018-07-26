import numpy as np
import theano
import theano.tensor as tt

from delfi.inference.BaseInference import BaseInference
from delfi.neuralnet.Trainer import Trainer
from delfi.neuralnet.LinearNet import LinearNet
from delfi.kernel.Kernel_learning import kernel_opt
from delfi.kernel.ImproperFlat import ImproperFlat
from delfi.kernel.BaseKernel import BaseKernel
from delfi.kernel.Gauss import Gauss
from delfi.kernel.Uniform import Uniform
from delfi.kernel.Epanechnikov import Epanechnikov

from delfi.neuralnet.LinearNet import LinearNet


dtype = theano.config.floatX


class SLPE(BaseInference):
    def __init__(self, generator, obs, prior_norm=False,
                 pilot_samples=100, convert_to_T=None, centre_on_obs=True,
                 reg_lambda=0.01, prior_mixin=0., seed=None, verbose=True, cbkrnl=None,
                 reinit_weights=False, **kwargs):
        """Sequential linear posterior estimation (SLPE)

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
            in each round            
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
        cbkrnl: calibration kernel

        Attributes
        ----------
        observables : dict
            Dictionary containing theano variables that can be monitored while
            training the neural network.
        """
        assert centre_on_obs # not meaningful otherwise. Change at own risk!

        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()
        self.verbose = verbose

        # bind generator, reset proposal attribute
        self.generator = generator

        # generate a sample to get input and output dimensions
        params, stats, _ = generator.gen(1, skip_feedback=True, verbose=False)
        kwargs.update({'n_inputs': stats.shape[1],
                       'n_outputs': params.shape[1],
                       'seed': self.gen_newseed()})

        self.network = LinearNet(**kwargs)
        self.kwargs = kwargs

        # parameters for z-transform of params
        if prior_norm:
            # z-transform for params based on prior
            self.params_mean = self.generator.prior.mean
            self.params_std = self.generator.prior.std
        else:
            # parameters are set such that z-transform has no effect
            self.params_mean = np.zeros((params.shape[1],))
            self.params_std = np.ones((params.shape[1],))

        # parameters for z-transform for stats
        if pilot_samples is not None and pilot_samples != 0:
            # determine via pilot run
            self.pilot_run(pilot_samples)
        else:
            # parameters are set such that z-transform has no effect
            self.stats_mean = np.zeros((stats.shape[1],))
            self.stats_std = np.ones((stats.shape[1],))

        self.obs = obs
        if centre_on_obs: 
            self.centre_on_obs()
        self.obz = self.standardize_stats(self.obs)

        self.round = 0
        self.convert_to_T = convert_to_T
        self.prior_mixin = 0. if prior_mixin is None else prior_mixin

        if cbkrnl is None: 
            self.cbkrnl = ImproperFlat(self.obz) 
        elif issubclass(cbkrnl, BaseKernel):
            self.cbkrnl = cbkrnl
        else:
            raise NotImplementedError

    def run(self, n_train=100, n_rounds=2, 
            kernel_loss=None, stop_on_nan=False,
            epochs_cbk=None, cbk_feature_layer=0, minibatch_cbk=None, 
            reg_lambdas=None, cbk_delta=None, naive_weights=False, **kwargs):
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

            # if round > 1, set new proposal distribution before sampling
            if self.round > 1:
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

            if type(epochs_cbk) == list:
                try:
                    epochs_cbk_round = epochs_cbk[self.round-1]
                except:
                    epochs_cbk_round = epochs_cbk[-1]
            else:
                epochs_cbk_round = epochs_cbk

            # draw training data (z-transformed params and stats)
            verbose = '(round {}) '.format(self.round) if self.verbose else False
            trn_data = self.gen(n_train_round, prior_mixin=self.prior_mixin, verbose=verbose)
            n_train_round = trn_data[0].shape[0]

            # precompute importance weights
            iws = np.ones((n_train_round,))

            cbkrnl, cbl = None, None
            idx_proposal = np.where(trn_data[2])[0]

            if not minibatch_cbk is None:
                minibatch_cbk = np.min((minibatch_cbk, idx_proposal.size))

            fstats = trn_data[1][idx_proposal,:].reshape(idx_proposal.size,-1)
            fobs_z = self.standardize_stats(self.obs).reshape(1,-1)

            if self.round==1 and cbk_delta is not None:
                print('setting initial kernel')
                delta = self.get_kernelwidth(cbk_delta, trn_data)
                self.cbkrnl = Uniform(self.obz.reshape(1,-1), spherical=True, bandwidth=delta)


            if self.generator.proposal is not None:
                params = self.params_std * trn_data[0] + self.params_mean
                p_prior = self.generator.prior.eval(params, log=False)
                p_proposal = self.generator.proposal.eval(params, log=False)
                iws *= p_prior / (self.prior_mixin * p_prior + (1. - self.prior_mixin) * p_proposal)

                # train calibration kernel (learns own normalization)
                if not kernel_loss is None:
                    if verbose:
                        print('fitting calibration kernel ...')

                    cbkrnl, cbl = kernel_opt(
                        iws=iws[idx_proposal].astype(np.float32), 
                        stats=fstats,
                        obs=fobs_z, 
                        kernel_loss=kernel_loss, 
                        epochs=epochs_cbk_round,
                        minibatch=minibatch_cbk,
                        stop_on_nan=stop_on_nan,
                        seed=self.gen_newseed(), 
                        **kwargs)
                    if verbose: 
                        print('done.')

                    self.cbkrnl = cbkrnl
            fstats = trn_data[1].reshape(n_train_round,-1)                    
            iws *= self.cbkrnl.eval(fstats)

            if naive_weights:
                iws = np.ones((n_train_round,))            

            logs.append({'cbkrnl' : self.cbkrnl, 'cbk_loss' : cbl})

            # normalize weights
            iws = (iws/np.sum(iws))*n_train_round
            trn_data = (trn_data[0], trn_data[1], iws)
            trn_datasets.append(trn_data)

            self.network.fit(trn_data, self.obz)

            posteriors.append(self.predict(self.obs))

        return logs, trn_datasets, posteriors


    def standardize_stats(self, x):

        return (x - self.stats_mean) / self.stats_std

    def predict(self, x):
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        """
        x_zt = self.standardize_stats(x)
        posterior = self.network.get_mog(x_zt)
        return posterior.ztrans_inv(self.params_mean, self.params_std)

    def get_kernelwidth(self, pdelta, trn_data):

        assert pdelta <= 1.

        dx = trn_data[1] - self.obz
        dist = np.sqrt(np.sum( dx**2, axis=1 ))
        idx = np.argsort( dist )[int(dist.size*pdelta)]        

        return dist[idx]
