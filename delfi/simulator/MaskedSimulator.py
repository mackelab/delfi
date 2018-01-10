import abc
import numpy as np

from delfi.utils.meta import ABCMetaDoc
from delfi.utils.progress import no_tqdm, progressbar

from delfi.simulator import BaseSimulator


class MaskedSimulator(BaseSimulator):
    def __init__(self, sim, mask, obs, seed=None):
        """Simulator with masked parameters

        This is a wrapper around BaseSimulator which imputes
        fixed values for specified parameters, reducing the 
        dimensionality of the problem.

        Parameters
        ----------
        sim : BaseSimulator
            The original simulator
        mask : 1d array 
            Boolean array determining the values to be imputed. False corresponds to imputed entries.
        obs : 1d array
            Array of parameters from which to impute the values
        seed : int or None
            See BaseSimulator
        """
        assert len(mask) == sim.dim_param, "Mask for simulator has incorrect length"

        super().__init__(dim_param=np.count_nonzero(mask), seed=seed)
        self.sim = sim

        self.mask = mask
        self.obs = obs

    def gen_single(self, params):
        real_params = self.obs.copy()
        real_params[self.mask] = params
        return self.sim.gen_single(real_params)

