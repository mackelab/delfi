import numpy as np
import scipy.misc
import scipy.stats

from delfi.distribution.TransformedNormal import TransformedNormal
from delfi.distribution.mixture.BaseMixture import BaseMixture

class MoTG(BaseMixture):
    def __init__(
            self,
            a,
            ms=None,
            Ps=None,
            Us=None,
            Ss=None,
            xs=None,
            upper=None, 
            lower=None, 
            flags=None,
            seed=None):
        """Mixture of 'transformed' Gaussians

        Creates a MoG with a valid combination of parameters or an already given
        list of log-, logit- and/or un-transformed Gaussian variables.

        Parameters
        ----------
        a : list or np.array, 1d
            Mixing coefficients
        ms : list, length n_components
            Means
        Ps : list, length n_components
            Precisions
        Us : list, length n_components
            Precision factors such that U'U = P
        Ss : list, length n_components
            Covariances
        xs : list, length n_components
            List of gaussian variables
        flags: list or np.array, 1d
            List of flags for each variable whether it is untransformed (=0),
            log-transformed (=1) or logit-transformed (=2).
        lower: list or np.array, 1d
            Lower bounds for logit-box. Defaults to 0 for each parameter.
        upper: list or np.array, 1d
            Upper bounds for logit-box. Defaults to 1 for each parameter.

        seed : int or None
            If provided, random number generator will be seeded
        """

        m = ms[0] if not ms is None else xs[0].m

        self.lower = np.zeros_like(m) if lower is None else np.atleast_1d(lower)
        self.upper = np.ones_like(m)  if upper is None else np.atleast_1d(upper)

        assert self.lower.ndim == self.upper.ndim
        assert self.lower.ndim == 1            

        self.flags = np.zeros_like(m) if flags is None else np.atleast_1d(flags)

        assert self.flags.ndim == 1            
        assert np.all(np.in1d(np.unique(self.flags), np.arange(3)))

        if ms is not None:
            super().__init__(
                a=np.asarray(a),
                ncomp=len(ms),
                ndim=len(ms[0]),
                seed=seed)

            if Ps is not None:
                self.xs = [
                    TransformedNormal(
                        m=m, P=P, upper=self.upper, lower=self.lower, 
                        flags=self.flags,
                        seed=self.gen_newseed()) for m, P in zip(
                        ms, Ps)]

            elif Us is not None:
                self.xs = [
                    TransformedNormal(
                        m=m, U=U,  upper=self.upper, lower=self.lower, 
                        flags=self.flags,
                        seed=self.gen_newseed()) for m, U in zip(
                        ms, Us)]

            elif Ss is not None:
                self.xs = [
                    TransformedNormal(
                        m=m, S=S,  upper=self.upper, lower=self.lower, 
                        flags=self.flags, 
                        seed=self.gen_newseed()) for m, S in zip(
                        ms, Ss)]

            else:
                raise ValueError('Precision information missing')

        elif xs is not None:
            super().__init__(
                a=np.asarray(a),
                ncomp=len(xs),
                ndim=xs[0].ndim,
                seed=seed)
            self.xs = xs

            # MoGT enforces its flags and bounds (initialize accordingly!):
            for x in xs:
                assert np.all(x.flags == self.flags)
                assert np.all(x.upper == self.upper)
                assert np.all(x.lower == self.lower)

        else:
            raise ValueError('Mean information missing')

    def _f(self, x):
        """ forward transformation
            x[i] -> x[i]        for Gaussian x[i]
            x[i] -> log(x[i])   for log-Normal x[i]
            x[i] -> logit(x[i]) for logit-Normal x[i]
         """
        return self.xs[0]._f(x)

    def _finv(self, x):
        """ backward transformation
            x[i] -> x[i]        for Gaussian x[i]
            x[i] -> exp(x[i])   for log-Normal x[i]
            x[i] -> expit(x[i]) for logit-Normal x[i]
         """
        return self.xs[0]._finv(x)


    @property
    def mean(self):
        """Means - not analytic for logit-normal!"""
        return np.nan * np.ones_like(self.m).reshape(-1)

    @property
    def std(self):
        """Standard deviations of marginals - not analytic for logit-normal!"""
        return np.nan * np.ones_like(np.diag(self.S)).reshape(-1)

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
        # See BaseMixture.py for docstring
        ps = np.array([c.eval(x, ii, log) for c in self.xs]).T
        res = scipy.misc.logsumexp(
            ps +
            np.log(
                self.a),
            axis=1) if log else np.dot(
            ps,
            self.a)

        return res

    @copy_ancestor_docstring
    def gen(self, n_samples=1):
        # See BaseMixture.py for docstring
        ii = self.gen_comp(n_samples)  # n_samples,

        ns = [np.sum((ii == i).astype(int)) for i in range(self.n_components)]
        samples = [x.gen(n) for x, n in zip(self.xs, ns)]
        samples = np.concatenate(samples, axis=0)
        self.rng.shuffle(samples)

        return samples
