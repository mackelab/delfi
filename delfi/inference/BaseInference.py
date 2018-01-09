import abc
import numpy as np

from delfi.neuralnet.NeuralNet import NeuralNet
from delfi.neuralnet.Trainer import Trainer
from delfi.utils.meta import ABCMetaDoc


class BaseInference(metaclass=ABCMetaDoc):
    def __init__(self, generator, prior_norm=True, pilot_samples=100,
                 seed=None, verbose=True, **kwargs):
        """Abstract base class for inference algorithms

        Inference algorithms must at least implement abstract methods of this
        class.

        Parameters
        ----------
        generator : generator instance
            Generator instance
        prior_norm : bool
            If set to True, will z-transform params based on mean/std of prior
        pilot_samples : None or int
            If an integer is provided, a pilot run with the given number of
            samples is run. The mean and std of the summary statistics of the
            pilot samples will be subsequently used to z-transform summary
            statistics.
        seed : int or None
            If provided, random number generator will be seeded
        kwargs : additional keyword arguments
            Additional arguments used when creating the NeuralNet instance

        Attributes
        ----------
        observables : dict
            Dictionary containing theano variables that can be monitored while
            training the neural network.
        """
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()
        self.verbose = verbose

        # bind generator, reset proposal attribute
        self.generator = generator
        self.generator.proposal = None

        # generate a sample to get input and output dimensions
        params, stats = generator.gen(1, skip_feedback=True, verbose=False)
        kwargs.update({'n_inputs': stats.shape[1:],
                       'n_outputs': params.shape[1],
                       'seed': self.gen_newseed()})

        self.network = NeuralNet(**kwargs)
        self.svi = self.network.svi

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

        # observables contains vars that can be monitored during training
        self.compile_observables()

    @abc.abstractmethod
    def loss(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    def gen(self, n_samples, n_reps=1, prior_mixin=0, verbose=None):
        """Generate from generator and z-transform

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        n_reps : int
            Number of repeats per parameter
        verbose : None or bool or str
            If None is passed, will default to self.verbose
        """
        verbose = self.verbose if verbose is None else verbose
        params, stats = self.generator.gen(n_samples, prior_mixin=prior_mixin, verbose=verbose)

        # z-transform params and stats
        params = (params - self.params_mean) / self.params_std
        stats = (stats - self.stats_mean) / self.stats_std

        return params, stats

    def gen_newseed(self):
        """Generates a new random seed"""
        if self.seed is None:
            return None
        else:
            return self.rng.randint(0, 2**31)

    def pilot_run(self, n_samples):
        """Pilot run in order to find parameters for z-scoring stats
        """
        verbose = '(pilot run) ' if self.verbose else False
        params, stats = self.generator.gen(n_samples, verbose=verbose)
        self.stats_mean = np.nanmean(stats, axis=0)
        self.stats_std = np.nanstd(stats, axis=0)

    def predict(self, x, deterministic=True):
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        deterministic : bool
            if True, mean weights are used for Bayesian network
        """
        x_zt = (x - self.stats_mean) / self.stats_std
        posterior = self.network.get_mog(x_zt, deterministic=deterministic)
        return posterior.ztrans_inv(self.params_mean, self.params_std)

    def compile_observables(self):
        """Creates observables dict"""
        self.observables = {}
        self.observables['loss.lprobs'] = self.network.lprobs
        for p in self.network.aps:
            self.observables[str(p)] = p

    def monitor_dict_from_names(self, monitor=None):
        """Generate monitor dict from list of variable names"""
        if monitor is not None:
            observe = {}
            if isinstance(monitor, str):
                monitor = [monitor]
            for m in monitor:
                if m in self.observables:
                    observe[m] = self.observables[m]
        else:
            observe = None
        return observe
