import delfi.distribution as dd
import numpy as np

from delfi.simulator.BaseSimulator import BaseSimulator


class Gauss(BaseSimulator):
    def __init__(self, dim=1, noise_cov=0.1, seed=None):
        """Gauss simulator

        Toy model that draws data from a distribution centered on theta with
        fixed noise.

        Parameters
        ----------
        dim : int
            Number of dimensions of parameters
        noise_cov : float
            Covariance of noise on observations
        seed : int or None
            If set, randomness is seeded
        """
        super().__init__(dim_param=dim, seed=seed)
        self.noise_cov = noise_cov*np.eye(dim)

    @copy_ancestor_docstring
    def gen_single(self, param):
        # See BaseSimulator for docstring
        param = np.asarray(param).reshape(-1)
        assert param.ndim == 1
        assert param.shape[0] == self.dim_param

        sample = dd.Gaussian(m=param, S=self.noise_cov,
                             seed=self.gen_newseed()).gen(1)

        return {'data': sample.reshape(-1)}
