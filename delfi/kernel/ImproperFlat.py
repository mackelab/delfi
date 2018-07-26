import numpy as np

from delfi.kernel.BaseKernel import BaseKernel


class ImproperFlat(BaseKernel):
    @staticmethod
    def kernel(u):
        u = np.atleast_1d(u)
        return np.ones(u.shape[0])/u.shape[0]
