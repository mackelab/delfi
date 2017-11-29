import numpy as np
import scipy.special

from delfi.distribution.BaseDistribution import BaseDistribution


class StudentsT(BaseDistribution):
    def __init__(self, m, S, dof, seed=None):
        """Student's T distribution

        Parameters
        ----------
        m : list or np.array, 1d
            Mean
        S : list or np.array, 1d
            Covariance
        dof : int
            Degrees of freedom
        seed : int or None
            If provided, random number generator will be seeded
        """
        m = np.asarray(m)
        self.m = m
        self.dof = dof
        assert(dof > 0)

        S = np.asarray(S)
        self.P = np.linalg.inv(S)
        self.C = np.linalg.cholesky(S).T  # C is upper triangular here
        self.S = S
        self.Pm = np.dot(self.P, m)
        self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))
        super().__init__(ndim=m.size, seed=seed)

    @property
    def mean(self):
        """Means"""
        return self.m.reshape(-1)

    @property
    def std(self):
        """Standard deviations of marginals"""
        return np.sqrt((self.dof / (self.dof - 2)) *
                       np.diag(self.S)).reshape(-1)

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
        # See BaseDistribution.py for docstring
        if ii is not None:
            raise NotImplementedError

        xm = x - self.m
        lp = np.log(1 + np.sum(np.dot(xm, self.P) * xm, axis=1) / self.dof)
        lp *= -(self.dof + self.ndim) / 2.0
        lp += np.log(scipy.special.gamma((self.dof + self.ndim) / 2))
        lp -= scipy.special.gammaln(self.dof / 2) + \
            self.ndim / 2 * np.log(self.dof * np.pi) - 0.5 * self.logdetP

        res = lp if log else np.exp(lp)
        return res

    @copy_ancestor_docstring
    def gen(self, n_samples=1):
        # See BaseDistribution.py for docstring
        u = self.rng.chisquare(self.dof, n_samples) / self.dof
        y = self.rng.multivariate_normal(np.zeros(self.ndim),
                                          self.S, (n_samples,))
        return self.m + y / np.sqrt(u)[:, None]
