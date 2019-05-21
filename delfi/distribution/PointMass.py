import numpy as np
from delfi.distribution.BaseDistribution import BaseDistribution

class PointMass(BaseDistribution):
    
    def __init__(self, loc, seed=None):
        """Discrete distribution

        Parameters
        ----------
        loc : np.array
            Position of the point mass in sampling space
        seed : int or None
            If provided, random number generator will be seeded
        """
        self.loc = np.asarray(loc).flatten()

        super().__init__(ndim=self.loc.size, seed=seed)
    
    @property
    def mean(self):
        """Means"""
        return self.loc

    @property
    def std(self):
        """Standard deviations of marginals"""
        return np.zeros(self.ndim)

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
        assert ii is None
        pp = np.prod(x==self.loc.reshape(1,-1),axis=1)
        return np.log(pp) if log else pp # this happily returns -inf
        
    @copy_ancestor_docstring
    def gen(self, n_samples=1, seed=None):
        return np.tile(self.loc.reshape(1,-1), (n_samples,1))
