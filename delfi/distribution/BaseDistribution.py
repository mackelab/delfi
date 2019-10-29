import abc
import numpy as np

from delfi.utils.meta import ABCMetaDoc


class BaseDistribution(metaclass=ABCMetaDoc):
    def __init__(self, ndim, seed=None):
        """Abstract base class for distributions

        Distributions must at least implement abstract properties and methods of
        this class.

        Parameters
        ----------
        ndim : int
            Number of ndimensions of the distribution
        seed : int or None
            If provided, random number generator will be seeded
        """
        self.ndim = ndim

        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

    @abc.abstractproperty
    def mean(self):
        """Means"""
        pass

    @abc.abstractproperty
    def std(self):
        """Standard deviations of marginals"""
        pass

    @abc.abstractmethod
    def eval(self, x, ii=None, log=True):
        """Method to evaluate pdf

        Parameters
        ----------
        x : int or list or np.array
            Rows are inputs to evaluate at
        ii : list
            A list of indices specifying which marginal to evaluate.
            If None, the joint pdf is evaluated
        log : bool, defaulting to True
            If True, the log pdf is evaluated

        Returns
        -------
        scalar
        """
        pass

    @abc.abstractmethod
    def gen(self, n_samples=1):
        """Method to generate samples

        Parameters
        ----------
        n_samples : int
            Number of samples to generate

        Returns
        -------
        n_samples x self.ndim
        """
        pass

    def reseed(self, seed):
        """Reseeds the distribution's RNG"""
        self.rng.seed(seed=seed)
        self.seed = seed

    def gen_newseed(self):
        """Generates a new random seed"""
        if self.seed is None:
            return None
        else:
            return self.rng.randint(0, 2**31)
