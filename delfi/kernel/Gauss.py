import numpy as np

from delfi.kernel.BaseKernel import BaseKernel


class Gauss(BaseKernel):
    @staticmethod
    def kernel(u):
        return 1/np.sqrt(2*np.pi)*np.exp(-0.5*u**2)
