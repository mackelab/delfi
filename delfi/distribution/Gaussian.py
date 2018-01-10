import numpy as np
import scipy.stats

from delfi.distribution.BaseDistribution import BaseDistribution
from delfi.distribution.StudentsT import StudentsT


class Gaussian(BaseDistribution):
    def __init__(self, m=None, P=None, U=None, S=None, Pm=None, seed=None):
        """Gaussian distribution

        Initialize a gaussian pdf given a valid combination of its parameters.
        Valid combinations are: m-P, m-U, m-S, Pm-P, Pm-U, Pm-S

        Focus is on efficient multiplication, division and sampling.

        Parameters
        ----------
        m : list or np.array, 1d
            Mean
        P : list or np.array, 2d
            Precision
        U : list or np.array, 2d
            Upper triangular precision factor such that U'U = P
        S : list or np.array, 2d
            Covariance
        C : list or np.array, 2d
            Upper or lower triangular covariance factor, in any case S = C'C
        Pm : list or np.array, 1d
            Precision times mean such that P*m = Pm
        seed : int or None
            If provided, random number generator will be seeded
        """
        assert m is None or np.asarray(m).ndim == 1
        assert P is None or np.asarray(P).ndim == 2
        assert U is None or np.asarray(U).ndim == 2
        assert S is None or np.asarray(S).ndim == 2
        assert Pm is None or np.asarray(Pm).ndim == 1

        self.__div__ = self.__truediv__
        self.__idiv__ = self.__itruediv__

        if m is not None:
            m = np.asarray(m)
            self.m = m
            ndim = self.m.size

            if P is not None:
                P = np.asarray(P)
                L = np.linalg.cholesky(P)  # P=LL' (lower triag)
                self.P = P
                self.C = np.linalg.inv(L)  # C is lower triangular here
                # S = C'C = L^{-1}^T L^{-1} = (LL^T)^{-1}
                self.S = np.dot(self.C.T, self.C)
                self.Pm = np.dot(P, m)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

            elif U is not None:
                U = np.asarray(U)
                self.P = np.dot(U.T, U)
                self.C = np.linalg.inv(U.T)  # C is lower triangular here
                self.S = np.dot(self.C.T, self.C)
                self.Pm = np.dot(self.P, m)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

            elif S is not None:
                S = np.asarray(S)
                self.P = np.linalg.inv(S)
                self.C = np.linalg.cholesky(S).T  # C is upper triangular here
                self.S = S
                self.Pm = np.dot(self.P, m)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            else:
                raise ValueError('Precision information missing')

        elif Pm is not None:
            Pm = np.asarray(Pm)
            self.Pm = Pm
            ndim = self.Pm.size

            if P is not None:
                P = np.asarray(P)
                L = np.linalg.cholesky(P)
                # L = np.linalg.cholesky(P + 0.001*np.identity(P.shape[0]))
                self.P = P
                self.C = np.linalg.inv(L)
                self.S = np.dot(self.C.T, self.C)
                self.m = np.linalg.solve(P, Pm)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

            elif U is not None:
                U = np.asarray(U)
                self.P = np.dot(U.T, U)
                self.C = np.linalg.inv(U.T)
                self.S = np.dot(self.C.T, self.C)
                self.m = np.linalg.solve(self.P, Pm)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

            elif S is not None:
                S = np.asarray(S)
                self.P = np.linalg.inv(S)
                self.C = np.linalg.cholesky(S).T
                self.S = S
                self.m = np.dot(S, Pm)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            else:
                raise ValueError('Precision information missing')

        else:
            raise ValueError('Mean information missing')

        super().__init__(ndim, seed=seed)

    @property
    def mean(self):
        """Means"""
        return self.m.reshape(-1)

    @property
    def std(self):
        """Standard deviations of marginals"""
        return np.sqrt(np.diag(self.S)).reshape(-1)

    def __mul__(self, other):
        """Multiply with another Gaussian"""
        assert isinstance(other, Gaussian)

        P = self.P + other.P
        Pm = self.Pm + other.Pm

        return Gaussian(P=P, Pm=Pm, seed=self.seed)

    def __imul__(self, other):
        """Incrementally multiply with another Gaussian"""
        assert isinstance(other, Gaussian)

        res = self * other

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def __truediv__(self, other):
        """Divide by another Gaussian

        Note that the resulting Gaussian might be improper."""
        assert isinstance(other, Gaussian)

        P = self.P - other.P
        Pm = self.Pm - other.Pm

        return Gaussian(P=P, Pm=Pm, seed=self.seed)

    def __itruediv__(self, other):
        """Incrementally divide by another Gaussian

        Note that the resulting Gaussian might be improper."""
        assert isinstance(other, Gaussian)

        res = self / other

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def __pow__(self, power, modulo=None):
        """Raise Gaussian to a power and get another Gaussian"""
        P = power * self.P
        Pm = power * self.Pm

        return Gaussian(P=P, Pm=Pm, seed=self.seed)

    def __ipow__(self, power):
        """Incrementally raise gaussian to a power"""
        res = self ** power

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def convert_to_T(self, dof):
        """Converts Gaussian to Student T

        Parameters
        ----------
        dof : int
            Degrees of freedom
        """
        return StudentsT(self.m, self.S, dof, seed=self.seed)

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
        # See BaseDistribution.py for docstring
        if ii is None:
            xm = x - self.m
            lp = -np.sum(np.dot(xm, self.P) * xm, axis=1)
            lp += self.logdetP - self.ndim * np.log(2.0 * np.pi)
            lp *= 0.5

        else:
            m = self.m[ii]
            S = self.S[ii][:, ii]
            if np.linalg.matrix_rank(S)==len(S[:,0]):
                lp = scipy.stats.multivariate_normal.logpdf(x, m, S, allow_singular=True)
                lp = np.array([lp]) if x.shape[0] == 1 else lp
            else:
                raise ValueError('Rank deficiency in covariance matrix')

        res = lp if log else np.exp(lp)
        return res

    @copy_ancestor_docstring
    def gen(self, n_samples=1):
        # See BaseDistribution.py for docstring
        z = self.rng.randn(n_samples, self.ndim)
        samples = np.dot(z, self.C) + self.m
        return samples

    def kl(self, other):
        """Calculates the KL divergence from this to another Gaussian

        Direction of KL is KL(this | other)
        """
        assert isinstance(other, Gaussian)
        assert self.ndim == other.ndim

        t1 = np.sum(other.P * self.S)

        m = other.m - self.m
        t2 = np.dot(m, np.dot(other.P, m))

        t3 = self.logdetP - other.logdetP

        t = 0.5 * (t1 + t2 + t3 - self.ndim)

        return t

    def ztrans_inv(self, mean, std):
        """Z-transform inverse

        Parameters
        ----------
        mean : array
            Mean vector
        std : array
            Std vector

        Returns
        -------
        Gaussian distribution
        """
        m = std*self.m + mean
        S = np.outer(std, std) * self.S
        return Gaussian(m=m, S=S, seed=self.seed)
