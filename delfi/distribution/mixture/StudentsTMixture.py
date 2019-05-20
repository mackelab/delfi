import numpy as np
import scipy.special
import scipy.misc
import scipy.stats

from delfi.distribution.StudentsT import StudentsT
from delfi.distribution.mixture.BaseMixture import BaseMixture


class MoT(BaseMixture):
    def __init__(self, a, ms=None, Ss=None, dofs=None, xs=None, seed=None):
        """Mixture of Student's T distributions

        Creates a MoT with a valid combination of parameters or an already given
        list of gaussian variables.

        Parameters
        ----------
        a : list or 1d array
            Mixing coefficients
        ms : list of length n_components
            Means
        Ss : list of length n_components
            Covariances
        dofs: list of length n_components
            Degrees of freedom
        xs : list of length n_components
            List of Student's T distributions
        seed : int or None
            If provided, random number generator will be seeded
        """
        if ms is not None:
            super().__init__(
                a=np.asarray(a),
                ncomp=len(ms),
                ndim=np.asarray(
                    ms[0]).ndim,
                seed=seed)
            self.xs = [
                StudentsT(
                    m=m,
                    S=S,
                    dof=dof,
                    seed=self.gen_newseed()) for m,
                S,
                dof in zip(
                    ms,
                    Ss,
                    dofs)]
        elif xs is not None:
            super().__init__(
                a=np.asarray(a),
                ncomp=len(xs),
                ndim=xs[0].ndim,
                seed=seed)
            self.xs = xs
        else:
            raise ValueError('Mean information missing')

    @copy_ancestor_docstring
    def gen(self, n_samples=1):
        # See BaseMixture.py for docstring
        ii = self.gen_comp(n_samples)  # n_samples,

        ns = [np.sum((ii == i).astype(int)) for i in range(self.n_components)]
        samples = [x.gen(n) for x, n in zip(self.xs, ns)]
        samples = np.concatenate(samples, axis=0)
        self.rng.shuffle(samples)

        return samples

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
        # See BaseMixture.py for docstring
        if ii is not None:
            raise NotImplementedError

        ps = np.array([c.eval(x, ii=None, log=log) for c in self.xs]).T
        res = scipy.special.logsumexp(
            ps +
            np.log(
                self.a),
            axis=1) if log else np.dot(
            ps,
            self.a)

        return res
