import numpy as np

from delfi.distribution.BaseDistribution import BaseDistribution


class Logistic(BaseDistribution):
    def __init__(self, mu=0.0, s=1.0, seed=None):
        """Distribution with independent dimensions and logistic marginals

        Parameters
        ---------
        mu : list, or np.array, 1d
            Means
        s : list, or np.array, 1d
            Scale factors
        seed : int or None
            If provided, random number generator will be seeded
        """
        mu, s = np.atleast_1d(mu), np.atleast_1d(s)
        if s.size == 1:
            s = np.full(mu.size, s[0])

        assert (s > 0).all() and np.isfinite(s).all() and np.isfinite(mu).all() and np.isreal(s).all() and \
               np.isreal(mu).all(), "bad params"
        assert s.ndim == 1 and mu.ndim == 1 and mu.size == s.size, "bad sizes"
        self.mu, self.s = mu, s

        super().__init__(ndim=mu.size, seed=seed)

    @property
    def mean(self):
        """Means"""
        return self.mu

    @property
    def std(self):
        """Standard deviations of marginals"""
        return self.s * np.pi / np.sqrt(3.0)

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
        # See BaseDistribution.py for docstring
        x = np.atleast_2d(x)
        assert x.shape[1] == self.ndim, "incorrect data dimension"

        if ii is None:
            ii = np.arange(self.ndim)

        z = (x - self.mu) / self.s
        logp_eachdim = -z - np.log(self.s) - 2.0 * np.log(1.0 + np.exp(-z))
        logp = logp_eachdim[:, ii].sum(axis=1)

        return logp if log else np.exp(logp)

    @copy_ancestor_docstring
    def gen(self, n_samples=1):
        # See BaseDistribution.py for docstring
        u = np.random.uniform(size=(n_samples, self.ndim))
        return self.mu + self.s * (np.log(u) - np.log(1 - u))
