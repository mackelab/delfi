import numpy as np

from .BaseDistribution import BaseDistribution


class MixedDistribution(BaseDistribution):
    """Mixed distribution (not to be confused with mixture distributions)

    This class defines a concatenation of a list of distributions. It supports `eval()` (without the ii option) and `gen()`.

    Parameters
    ----------
    dists : array of distributions
        Array of distributions
    seed : int or None
        If provided, random number generator will be seeded
    """
    def __init__(self, dists, seed=None):
  
        dists = [ d for d in dists if d.ndim > 0 ]
        self.dimlist = [ d.ndim for d in dists ]
        super().__init__(ndim=np.sum(self.dimlist), seed=seed)

        self.dists = dists

    def mean(self):
        return np.concatenate([d.mean for d in self.dists])

    def std(self):
        return np.concatenate([d.std for d in self.dists])

    def eval(self, x, ii=None, log=True):
        if ii != None:
            raise NotImplementedError

        xsplit = np.split(x, np.cumsum(self.dimlist), axis=-1)

        assert [ d.ndim == xd.shape[-1] for d, xd in zip(self.dists, xsplit) ]
        evallist = [ d.eval(xd) for d, x in zip(self.dists, xsplit) ]
        return np.prod(evallist, axis=0)

    def gen(self, n_samples=1):
        return np.concatenate([ d.gen(n_samples) for d in self.dists ], axis=-1)
