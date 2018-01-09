import abc
import numpy as np

from delfi.utils.meta import ABCMetaDoc
from delfi.utils.progress import no_tqdm, progressbar

from delfi.simulator import BaseSimulator

class MaskedSimulator(BaseSimulator):
    def __init__(self, sim, mask, obs, seed=None):
        assert len(mask) == sim.dim_param, "Mask for simulator has incorrect length"

        super().__init__(dim_param=np.count_nonzero(mask), seed=seed)
        self.sim = sim

        self.mask = mask
        self.obs = obs

    def gen_single(self, params):
        real_params = self.obs.copy()
        real_params[self.mask] = params
        return self.sim.gen_single(real_params)

