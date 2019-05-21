import numpy as np

from delfi.simulator.BaseSimulator import BaseSimulator


class Blob(BaseSimulator):
    def __init__(self, isize=32, maxval=255, sigma=None, seed=None,
                 xy_abs_max=17, gamma_min=0.2, gamma_max=5.05):
        """Gauss simulator

        Toy model that generates images containing a blob. For details, see
        figure 3 of https://arxiv.org/pdf/1805.09294.pdf

        Parameters
        ----------
        isize: int
            Number of image rows and columns
        maxval: int
            Maximum pixel value
        xy_abs_max: int
            Maximum distance of blob center from image center, in pixels
        gamma_min: float
            Parameter controlling blob shape
        gamma__max: float
            Parameter controlling blob shape
        sigma : float
            Sigma value. If none, it will become a 4th parameter for inference.
        seed : int or None
            If set, randomness is seeded
        """
        dim = 4 if sigma is None else 3
        super().__init__(dim_param=dim, seed=seed)
        self.isize, self.maxval = isize, maxval
        self.xy_abs_max, self.gamma_min, self.gamma_max = \
            xy_abs_max, gamma_min, gamma_max
        self.x, self.y = \
            np.meshgrid(np.linspace(-isize // 2, isize // 2, isize),
                        np.linspace(-isize // 2, isize // 2, isize))
        self.sigma = sigma

    @copy_ancestor_docstring
    def gen_single(self, params):
        # See BaseSimulator for docstring
        if self.sigma is None:
            assert params.size == 4
            xo, yo, gamma, sigma = params
        else:
            assert params.size == 3
            xo, yo, gamma = params
            sigma = self.sigma

        xo = self.xy_abs_max * (2.0 / (1.0 + np.exp(-xo)) - 1.0)
        yo = self.xy_abs_max * (2.0 / (1.0 + np.exp(-yo)) - 1.0)
        gamma = (self.gamma_max - self.gamma_min) / (1. + np.exp(-gamma)) \
            + self.gamma_min

        r = (self.x - xo) ** 2 + (self.y - yo) ** 2
        p = 0.1 + 0.8 * np.exp(-0.5 * (r / sigma ** 2) ** gamma)

        counts = self.rng.binomial(self.maxval, p) / self.maxval

        return {'data': counts.reshape(-1)}