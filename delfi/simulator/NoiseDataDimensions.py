import numpy as np
from copy import deepcopy

from delfi.simulator.BaseSimulator import BaseSimulator


class NoiseDataDimensions(BaseSimulator):
    def __init__(self, model, noise_dist, seed=None, deepcopy_inputs=True, rand_permute=True):
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
        super().__init__(dim_param=model.dim_param + noise_dist.ndim, seed=seed)
        if deepcopy_inputs:
            model, noise_dist = deepcopy(model), deepcopy(noise_dist)
        self.model, self.noise_dist = model, noise_dist

        if seed is not None:
            self.reseed(seed)
        if rand_permute:
            self.permutation = self.rng.permutation(self.dim_param)
        else:
            self.permutation = np.arange(self.dim_param)

    def reseed(self, seed):
        super().reseed(seed)
        self.model.reseed(self.gen_newseed())
        self.noise_dist.reseed(self.gen_newseed())

    @copy_ancestor_docstring
    def gen_single(self, param):
        # See BaseSimulator for docstring
        param = np.asarray(param).reshape(-1)
        assert param.ndim == 1
        assert param.shape[0] == self.dim_param

        model_sample = self.model.gen(1)
        noise_sample = self.noise_dist.gen(1)

        return {'data': np.concatenate((model_sample.reshape(-1), noise_sample))[self.permutation]}
