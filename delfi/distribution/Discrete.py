import numpy as np

from delfi.distribution.BaseDistribution import BaseDistribution


class Discrete(BaseDistribution):
    def __init__(self, p, seed=None):
        """Discrete distribution

        Parameters
        ----------
        p : list or np.array, 1d
            Probabilities of elements, must sum to 1
        seed : int or None
            If provided, random number generator will be seeded
        """
        super().__init__(ndim=1, seed=seed)

        p = np.asarray(p)
        assert p.ndim == 1, 'p must be a 1-d array'
        assert np.isclose(np.sum(p), 1), 'p must sum to 1'
        self.p = p

    @property
    def mean(self):
        """Means"""
        pass

    @property
    def std(self):
        """Standard deviations of marginals"""
        pass

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
        raise NotImplementedError("To be implemented")

    @copy_ancestor_docstring
    def gen(self, n_samples=1, seed=None):
        # See BaseDistribution.py for docstring
        c = np.cumsum(self.p[:-1])[np.newaxis, :]  # cdf
        r = self.rng.rand(n_samples, 1)
        return np.sum((r > c).astype(int), axis=1).reshape(-1, 1)
