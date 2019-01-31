import numpy as np
import scipy.stats

from delfi.distribution.BaseDistribution import BaseDistribution
#from delfi.distribution.Gaussian import Gaussian
from delfi.distribution.StudentsT import StudentsT


class LogNormal(BaseDistribution):
    def __init__(self, m=None, P=None, U=None, S=None, Pm=None, seed=None):
        """LogNormal distribution

        Initialize a lognormal pdf given a valid combination of its parameters.
        Valid combinations are: m-P, m-U, m-S, Pm-P, Pm-U, Pm-S

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
        return np.exp(self.m).reshape(-1)

    @property
    def std(self):
        """Standard deviations of marginals"""
        v = (np.exp(np.diag(self.S))-1)*np.exp(2*self.m+np.diag(self.S))
        return np.sqrt(v).reshape(-1)

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
        # See BaseDistribution.py for docstring        

        if x.ndim==1:
            x = x.reshape(-1,1)

        logx = np.log(x)

        if ii is None:
            xm = logx - self.m
            lp = -np.sum(np.dot(xm, self.P) * xm, axis=1)
            lp += self.logdetP - self.ndim * np.log(2.0 * np.pi)
            lp *= 0.5
            lp -= np.sum(logx,axis=1)

        else:
            m = self.m[ii]
            S = self.S[ii][:, ii]
            if np.linalg.matrix_rank(S)==len(S[:,0]):
                lp = scipy.stats.multivariate_normal.logpdf(logx, m, S, allow_singular=True)
                lp = np.array([lp]) if x.shape[0] == 1 else lp
                lp -= np.sum(logx,axis=1)
            else:
                raise ValueError('Rank deficiency in covariance matrix')

        res = lp if log else np.exp(lp)
        return res

    @copy_ancestor_docstring
    def gen(self, n_samples=1):
        # See BaseDistribution.py for docstring
        z = self.rng.randn(n_samples, self.ndim)
        samples = np.dot(z, self.C) + self.m
        return np.exp(samples)

    #def convert_to_exp(self):
    #    """Return equivalent Gaussian for exp(X)"""
    #    return Gaussian(m=self.m, P=self.P)