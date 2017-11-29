import numpy as np

from delfi.kernel.BaseKernel import BaseKernel


class Epanechnikov(BaseKernel):
    @staticmethod
    def kernel(u):
        if np.abs(u) <= 1:
            return 3/4*(1-u**2)
        else:
            return 0.
