import numpy as np
from delfi.distribution.BaseDistribution import BaseDistribution


class TransformedDistribution(BaseDistribution):
    """Distribution object that carries out an invertible change of variables
    for another distribution object

    Parameters
    ----------
    distribution : delfi distribution or mixture object
        Original distrib. to be transformed. Must implement eval() and gen()
    bijection : callable
        Bijective mapping from original distrib.'s random variable to this one's
    inverse_bijection: callable
        Inverse of the bijective mapping, going from this distribution's random
        variable to the original one's.
    bijection_log_determinant: callable
        Log determinant of the bijection from the original distribution's random
        variable to this one's.
    """
    def __init__(self, distribution, bijection, inverse_bijection,
                 bijection_log_determinant):
        self.distribution = distribution
        self.bijection, self.inverse_bijection = bijection, inverse_bijection
        self.bijection_log_determinant = bijection_log_determinant
        self.ndim = distribution.ndim

    @property
    def mean(self):
        """Means"""
        return np.nan(self.ndim)  # generally unknown

    @property
    def std(self):
        """Standard deviations of marginals"""
        return np.nan(self.ndim)  # generally unknown

    @copy_ancestor_docstring
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
        assert ii is None, "cannot marginalize transformed distributions"
        x_original = self.inverse_bijection(x)

    @copy_ancestor_docstring
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
        samples = self.distribution.gen(n_samples=n_samples)
        return self.bijection(samples)

    def reseed(self, seed):
        """Reseeds the distribution's RNG"""
        self.distribution.reseed(seed)

    def gen_newseed(self):
        """Generates a new random seed"""
        return self.distribution.gen_newseed()
