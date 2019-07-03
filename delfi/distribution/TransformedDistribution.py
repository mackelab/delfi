import numpy as np
from delfi.distribution.BaseDistribution import BaseDistribution


class TransformedDistribution(BaseDistribution):
    """Distribution object that carries out an invertible change of variables for another distribution object

    Parameters
    ----------
    distribution : delfi distribution or mixture object
        Original distribution to be transformed. Must implement eval and gen methods.
    bijection : callable
        Bijective mapping from original distribution's random variable to this one's.
    inverse_bijection: callable
        Inverse of the bijective mapping, going from this distribution's random variable to the original one's.
    bijection_log_determinant: callable
        Log determinant of the bijection from the original distribution's random variable to this one's.
    """
    def __init__(self, distribution, bijection, inverse_bijection, bijection_log_determinant):
        self.ndim = distribution.ndim

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
