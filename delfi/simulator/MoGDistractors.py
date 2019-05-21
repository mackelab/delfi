import delfi.distribution as dd
import numpy as np

from delfi.simulator.BaseSimulator import BaseSimulator


class MoGDistractors(BaseSimulator):
    def __init__(self, dim=2, noise_cov=1.0, distractors=10, p_true=None, n_samples=1, seed=None):
        """Gaussian Mixture simulator

        Toy model that draws data from a mixture distribution with 1 "moving" component that depends on the parameters,
        and several other "distractor" components that do not.

        Parameters
        ----------
        dim : int
            Number of dimensions of parameters
        noise_cov : float or dim X dim covariance matrix as array
            Covariance for the moving component
        distractors: int or MoG
            MoG components defining the distractors. Will be generated automatically an integer (count) is given.
        p_true: float
            Probability that each sampled data point is NOT from a distractor. If None, mixture weights are uniform.
        n_samples: int
            Number of data points per simulation, concatenated
        seed : int or None
            If set, randomness is seeded
        """
        super().__init__(dim_param=dim, seed=seed)
        self.n_samples = n_samples
        if type(noise_cov) is float:
            noise_cov = noise_cov * np.eye(dim)
        self.noise_cov = noise_cov
        if type(distractors) is int:
            self.a = np.ones(distractors) / distractors
            self.ms = np.random.rand(distractors, dim) * 20.0 - 10.0
            self.Ss = [0.1 + 0.9 * np.diag(np.random.rand(dim)) for _ in range(distractors)]
        else:
            assert isinstance(distractors, dd.MoG)
            self.a = distractors.a
            self.ms = [x.m for x in distractors.xs]
            self.Ss = [x.S for x in distractors.xs]
        if p_true is None:
            p_true = 1.0 / (self.a.size + 1.0)
        self.p_true = p_true

    @copy_ancestor_docstring
    def gen_single(self, param):
        # See BaseSimulator for docstring
        param = np.asarray(param).reshape(-1)
        assert param.ndim == 1
        assert param.shape[0] == self.dim_param

        q_moving = dd.Gaussian(m=param, S=self.noise_cov, seed=self.gen_newseed())
        q_distractors = dd.MoG(a=self.a, ms=self.ms, Ss=self.Ss, seed=self.gen_newseed())

        samples = []
        for _ in range(self.n_samples):
            if np.random.rand() < self.p_true:
                samples.append(q_moving.gen(1))
            else:
                samples.append(q_distractors.gen(1))

        return {'data': np.concatenate(samples, axis=0)}
