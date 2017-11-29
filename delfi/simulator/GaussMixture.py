import delfi.distribution as dd
import numpy as np

from delfi.simulator.BaseSimulator import BaseSimulator


class GaussMixture(BaseSimulator):
    def __init__(self, dim=1, noise_cov=[1.0, 0.1], seed=None):
        """Gaussian Mixture simulator

        Toy model that draws data from a mixture distribution with 2 components
        that have mean theta and fixed noise.

        Parameters
        ----------
        dim : int
            Number of dimensions of parameters
        noise_cov : list
            Covariance of noise on observations
        seed : int or None
            If set, randomness is seeded
        """
        super().__init__(dim_param=dim, seed=seed)
        self.a = [0.5, 0.5]  # mixture weights
        self.noise_cov = [nc*np.eye(dim) for nc in noise_cov]

    @copy_ancestor_docstring
    def gen_single(self, param):
        # See BaseSimulator for docstring
        param = np.asarray(param).reshape(-1)
        assert param.ndim == 1
        assert param.shape[0] == self.dim_param

        sample = dd.MoG(a=self.a, ms=[param for p in range(2)],
                        Ss=self.noise_cov, seed=self.gen_newseed()).gen(1)

        return {'data': sample.reshape(-1)}
