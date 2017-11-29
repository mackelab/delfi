import abc
import numpy as np

from delfi.utils.meta import ABCMetaDoc


class BaseKernel(metaclass=ABCMetaDoc):
    def __init__(self, obs, bandwidth=1., spherical=True, atleast=None):
        """Abstract base class for kernels

        Distributions must at least implement abstract methods of this class.

        Parameters
        ----------
        obs : 1 x dim
            center of kernel
        bandwidth : float
            bandwidth of kernel (isotropic)
        spherical : bool
            construct spherical or multiplicative kernel [1]
        atleast : float [0,1]
            if specified, kernel will default to a uniform kernel
            iff the fraction of weights is below the limit specified by atleast

        Notes
        -----
        See [2] for descriptions of common kernel functions.

        [1]: http://www.buch-kromann.dk/tine/nonpar/Nonparametric_Density_Estimation_multidim.pdf
        [2]: https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use
        """
        assert obs.shape[0] == 1, 'obs.shape[0] must be 1'
        assert obs.shape[1] >= 1, 'obs.shape[1] must be >= 1'
        if atleast is not None:
            assert 0 <= atleast <= 1, 'atleast must be in [0,1]'

        self.dim = obs.shape[1]
        self.obs = obs
        self.bandwidth = bandwidth
        self.spherical = spherical
        self.atleast = atleast

        self.H = bandwidth * np.eye(self.dim)
        self.invH = np.linalg.inv(self.H)
        self.detH = np.linalg.det(self.H)

        self.normalizer = 1. / self.kernel(0.)

    @abc.abstractmethod
    def kernel(u):
        pass

    def eval(self, x):
        """Kernel for loss calibration

        Parameters
        ----------
        x : N x dim
            points at which to evaluate kernel

        Returns
        -------
        weights : N
            normalized to be 1. for x = obs
        """
        assert x.shape[0] >= 1, 'x.shape[0] needs to be >= 1'
        assert x.shape[1] == self.dim, 'x.shape[1] needs to be == self.obs'

        out = np.ones((x.shape[0],))

        for n in range(x.shape[0]):
            us = np.dot(self.invH, np.array(x[n] - self.obs).T)

            if self.spherical:
                out[n] = self.normalizer * self.kernel(np.linalg.norm(us))
            else:
                for u in us:
                    out[n] *= self.normalizer * self.kernel(u)

        # check fraction of points accepted
        if self.atleast is not None:
            accepted = out > 0.0
            if sum(accepted) / len(accepted) < self.atleast:
                dists = np.linalg.norm(x - self.obs, axis=1)
                N = int(np.round(x.shape[0] * self.atleast))
                idx = np.argsort(dists)[:N]
                out = np.zeros((x.shape[0],))
                out[idx] = 1.
                return out

        return out
