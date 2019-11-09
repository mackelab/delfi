import numpy as np 
from scipy.stats import poisson
from delfi.distribution.BaseDistribution import BaseDistribution


class Poisson(BaseDistribution):
    def __init__(self, mu=0., offset=0., seed=None):
        """Univariate (!) Poisson distribution
        Parameters
        ----------
        mu: shape parameter of the Poisson (Poisson rate)
        offset: shift in the mean parameter, see scipy.stats.Poisson documentation. 
        seed : int or None
            If provided, random number generator will be seeded
        """
        super().__init__(ndim=1, seed=seed)
        
        mu = np.atleast_1d(mu)
        assert mu.ndim == 1, 'mu must be a 1-d array'
        assert offset >= 0, 'offset must not be negative'
        
        self.mu = mu
        self.offset = offset
        self._poisson = poisson(mu=mu, loc=offset)

    @property
    def mean(self):
        """Means"""
        return self.mu

    @property
    def std(self):
        """Standard deviations of marginals"""
        return np.sqrt(self.mu)

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
        # univariate distribution only, i.e. ii=[0] in any case
        assert ii is None, 'this is a univariate Poisson, ii must be None.'

        # x should have a second dim with length 1, not more
        x = np.atleast_2d(x)
        assert x.shape[1] == 1, 'x needs second dim'
        assert not x.ndim > 2, 'no more than 2 dims in x'

        res = self._poisson.logpmf(x) if log else self._poisson.pmf(x)
        # reshape to (nbatch, )
        return res.reshape(-1)

    @copy_ancestor_docstring
    def gen(self, n_samples=1, seed=None):
        # See BaseDistribution.py for docstring

        x = self._poisson.rvs(random_state=self.rng, size=(n_samples, self.ndim))
        return x
