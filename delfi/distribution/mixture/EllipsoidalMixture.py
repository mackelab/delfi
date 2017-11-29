import numpy as np
import scipy.special
import scipy.misc
import scipy.stats

from delfi.distribution.Gaussian import Gaussian
from delfi.distribution.mixture.BaseMixture import BaseMixture
from numpy import linalg as la
from scipy.special import gammaincinv


class MoE(BaseMixture):
    def __init__(self, a, ms=None, Ps=None, Us=None, Ss=None, xs=None,
                 seed=None, beta=0.99):
        """Mixture of Ellipsoidals

        Creates a MoE with a valid combination of parameters or an already given
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
        beta : float
            Mass to preserve when sampling
        """
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

        self.threshold = 2 * gammaincinv(0.5 * self.ndim, beta)

    @copy_ancestor_docstring
    def gen(self, n_samples=1):
        for i in range(len(self.xs)):
            self.xs[i].L = la.cholesky(self.xs[i].S)

        samp = []
        for _ in range(n_samples):
            samp.append(self.gen_single())
        samp = np.array(samp)

        return samp

    def gen_single(self):
        """Generate single sample
        """
        def draw_proposal(xs):
            μ = xs.m
            L = xs.L
            x = self.uni_rand_ellipse(L * np.sqrt(self.threshold))
            return x.ravel() + μ.ravel()

        while True:
            i = self.gen_comp(1)[0]  # component index
            x = draw_proposal(self.xs[i])
            ρ = np.zeros(self.ncomp)
            for j, xs in enumerate(self.xs):
                μ = xs.m
                L = xs.L
                z = la.solve(L, (x - μ))
                ρ[j] = np.dot(z, z)
            π = 1 / np.sum(ρ < self.threshold)
            if self.rng.rand() < π:
                return x

    def uni_rand_ellipse(self, L, n=1):
        """Sample from ellipsoid

        Parameters
        ----------
        L : np.array
            Cholesky factorization of covariance matrix s.t. Σ = LL'
        n : int
            number of samples to generate
        """
        m = L.shape[0]
        x = self.rng.normal(size=(m, n))

        # map the points on the n-dimensional hypersphere
        w = np.sqrt(np.sum(x ** 2, axis=0))  # norm of the vector
        x = x / w  # normalized vector

        # generate points distributed as m * r^(m-1) for 0 < r < 1
        u = self.rng.uniform(size=n)
        r = np.outer(np.ones(m), u) ** (1 / m)

        φsph = r * x  # φ is uniformely distributed within the unit sphere
        return np.dot(L, φsph)  # rescale the sphere into an ellipsoid

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
        # Returns 1 everywhere (unnormalized)
        if ii is not None:
            raise NotImplementedError

        ps = np.array([c.eval(x, ii, log) for c in self.xs]).T
        ps *= 0
        ps += 1.
        ps = ps.squeeze()

        if log:
            return np.log(ps)
        else:
            return ps
