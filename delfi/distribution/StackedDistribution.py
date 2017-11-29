
import numpy as np

from delfi.distribution.BaseDistribution import BaseDistribution

class StackedDistribution(BaseDistribution):
    def __init__(self, ps, seed=None):
        """Stacked multivariate distribution for flexible block distributions

        Blockwise independent distribution with variable blocks described by
        distribution objects.

        Parameters
        ----------
        ps : list or np.array, 1d
            Iterable with distribution objects
        seed : int or None
            If provided, random number generator will be seeded
        """

        #if ndims is not None:
        #    assert len(ndims) == len(ps)
        #    assert np.all( [ndims[i].size == ps[i].ndim for i in range(len(ps))])
        #    assert np.all(np.unique(ndims) == np.arange(np.sum(ndims)))
        #    self.ndims = ndims
        #else:
        
        i, self.ndims = 0, []
        for p in ps:
            self.ndims.append(np.arange(i, i+p.ndim))
            i += p.ndim

        super().__init__(ndim=i, seed=seed)
        self.ps = ps

    @property
    def mean(self):
        """Means"""
        #m = np.empty(self.ndim)
        return np.concatenate([p.mean for p in self.ps])

    @property
    def std(self):
        """Standard deviations of marginals"""
        return np.concatenate([p.std for p in self.ps])

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
            probs = np.hstack([np.asarray(self.pii(p,x,ndim,ii,log)).reshape(-1,1) for p, ndim in zip(self.ps, self.ndims)])
            return np.sum(probs,axis=1) if log else np.prod(probs,axis=1)

    @staticmethod
    def pii(p, x, ndim, ii, log):
        """Method call to index and evaluate individual block pdfs"""
        ii_ = np.where(np.in1d(ndim, ii))[0] if ii is not None else np.arange(ndim.size)
        ix_ = np.where(np.in1d(ii, ndim))[0] if ii is not None else np.array([ndim])        
        if len(ii_) > 0:
            if x.ndim==2:
                return p.eval(x[:,ix_], ii_, log)
            else: 
                assert len(ix_)==1 and ix_[0]==0
                return p.eval(x, ii_, log)
        else:
            return np.zeros(x.shape[0]) if log else np.ones(x.shape[0])

    @copy_ancestor_docstring
    def gen(self, n_samples=1, seed=None):
        # See BaseDistribution.py for docstring
        return np.hstack([p.gen(n_samples=n_samples) for p in self.ps])
