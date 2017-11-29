import numpy as np
from scipy.stats import gamma
from delfi.distribution.BaseDistribution import BaseDistribution


class Gamma(BaseDistribution):
    def __init__(self, alpha=1., beta=1., seed=None):
        """Univariate (!) Gamma distribution

        Parameters
        ----------
        alpha : list, or np.array, 1d
            Shape parameters
        beta : list, or np.array, 1d
            inverse scale paramters
        seed : int or None
            If provided, random number generator will be seeded
        """
        super().__init__(ndim=1, seed=seed)

        alpha, beta = np.atleast_1d(alpha), np.atleast_1d(beta)
        assert alpha.ndim == 1, 'alpha must be a 1-d array'
        assert alpha.size == beta.size, 'alpha and beta must match in size'
        assert np.all(alpha > 0.), 'Should be greater than zero.'
        assert np.all(beta > 0.), 'Should be greater than zero.'
        self.alpha = alpha
        self.beta = beta
        self._gamma = gamma(a=alpha, scale=1./beta)

    @property
    def mean(self):
        """Means"""
        return self.alpha / self.beta

    @property
    def std(self):
        """Standard deviations of marginals"""
        return np.sqrt( self.alpha ) / self.beta

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
        # univariate distribution only, i.e. ii=[0] in any case
        return self._gamma.logpdf(x) if log else self._gamma.pdf(x)

    @copy_ancestor_docstring
    def gen(self, n_samples=1, seed=None):
        # See BaseDistribution.py for docstring
        
        x = self.rng.gamma(shape=self.alpha, 
                           scale=1./self.beta, 
                           size=(n_samples, self.ndim))
        return x
