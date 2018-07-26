import numpy as np
import scipy.misc
import scipy.stats

from delfi.distribution.Gaussian import Gaussian
from delfi.distribution.mixture.BaseMixture import BaseMixture
from delfi.distribution.mixture.StudentsTMixture import MoT
from delfi.distribution.mixture.EllipsoidalMixture import MoE


class MoG(BaseMixture):
    def __init__(
            self,
            a,
            ms=None,
            Ps=None,
            Us=None,
            Ss=None,
            xs=None,
            seed=None):
        """Mixture of Gaussians

        Creates a MoG with a valid combination of parameters or an already given
        list of Gaussian variables.

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
        seed : int or None
            If provided, random number generator will be seeded
        """
        self.__div__ = self.__truediv__
        self.__idiv__ = self.__itruediv__

        if ms is not None:
            super().__init__(
                a=np.asarray(a),
                ncomp=len(ms),
                ndim=np.asarray(
                    ms[0]).ndim,
                seed=seed)

            if Ps is not None:
                self.xs = [
                    Gaussian(
                        m=m, P=P, seed=self.gen_newseed()) for m, P in zip(
                        ms, Ps)]

            elif Us is not None:
                self.xs = [
                    Gaussian(
                        m=m, U=U, seed=self.gen_newseed()) for m, U in zip(
                        ms, Us)]

            elif Ss is not None:
                self.xs = [
                    Gaussian(
                        m=m, S=S, seed=self.gen_newseed()) for m, S in zip(
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

        else:
            raise ValueError('Mean information missing')

    def __mul__(self, other):
        """Multiply with a single gaussian"""
        assert isinstance(other, Gaussian)

        ys = [x * other for x in self.xs]

        lcs = np.empty_like(self.a)

        for i, (x, y) in enumerate(zip(self.xs, ys)):

            lcs[i] = x.logdetP + other.logdetP - y.logdetP
            lcs[i] -= np.dot(x.m, np.dot(x.P, x.m)) + \
                np.dot(other.m, np.dot(other.P, other.m)) - \
                np.dot(y.m, np.dot(y.P, y.m))
            lcs[i] *= 0.5

        la = np.log(self.a) + lcs
        la -= scipy.misc.logsumexp(la)
        a = np.exp(la)

        return MoG(a=a, xs=ys, seed=self.seed)

    def __imul__(self, other):
        """Incrementally multiply with a single gaussian"""
        assert isinstance(other, Gaussian)

        res = self * other

        self.a = res.a
        self.xs = res.xs

        return res

    def __truediv__(self, other):
        """Divide by a single gaussian"""
        assert isinstance(other, Gaussian)

        ys = [x / other for x in self.xs]

        lcs = np.empty_like(self.a)

        for i, (x, y) in enumerate(zip(self.xs, ys)):

            lcs[i] = x.logdetP - other.logdetP - y.logdetP
            lcs[i] -= np.dot(x.m, np.dot(x.P, x.m)) - \
                np.dot(other.m, np.dot(other.P, other.m)) - \
                np.dot(y.m, np.dot(y.P, y.m))
            lcs[i] *= 0.5

        la = np.log(self.a) + lcs
        la -= scipy.misc.logsumexp(la)
        a = np.exp(la)

        return MoG(a=a, xs=ys, seed=self.seed)

    def __itruediv__(self, other):
        """Incrementally divide by a single gaussian"""
        assert isinstance(other, Gaussian)

        res = self / other

        self.a = res.a
        self.xs = res.xs

        return res

    def calc_mean_and_cov(self):
        """Calculate the mean vector and the covariance matrix of the MoG"""
        ms = [x.m for x in self.xs]
        m = np.dot(self.a, np.array(ms))

        msqs = [x.S + np.outer(mi, mi) for x, mi in zip(self.xs, ms)]
        S = np.sum(
            np.array([a * msq for a, msq in zip(self.a, msqs)]), axis=0) - np.outer(m, m)

        return m, S

    @property
    def mean(self):
        """Means"""
        return self.calc_mean_and_cov()[0].reshape(-1)

    @property
    def std(self):
        """Standard deviations of marginals"""
        return np.sqrt(np.diag(self.calc_mean_and_cov()[1])).reshape(-1)

    def convert_to_T(self, dofs):
        """Convert to Mixture of Student's T distributions

        Parameters
        ----------
        dofs : int or list of ints
            Degrees of freedom of component distributions
        """
        if type(dofs) == int:
            dofs = [dofs for i in range(len(self.xs))]
        ys = [x.convert_to_T(dof) for x, dof in zip(self.xs, dofs)]
        return MoT(self.a, xs=ys, seed=self.seed)

    def convert_to_E(self, beta=0.99):
        """Convert to Mixture of ellipsoidal distributions
        """
        return MoE(self.a, xs=self.xs, seed=self.seed, beta=beta)

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

    def project_to_gaussian(self):
        """Returns a gaussian with the same mean and precision as the mog"""
        m, S = self.calc_mean_and_cov()
        return Gaussian(m=m, S=S, seed=self.seed)

    def ztrans_inv(self, mean, std):
        """Z-transform inverse"""
        xs = [x.ztrans_inv(mean, std) for x in self.xs]
        return MoG(self.a, xs=xs, seed=self.seed)
