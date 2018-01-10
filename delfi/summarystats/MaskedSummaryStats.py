import abc
import numpy as np

from delfi.summarystats import BaseSummaryStats

class MaskedSummaryStats(BaseSummaryStats)
    def __init__(self, s, mask, obs, seed=None):
        super().__init__(seed)
        self.s = s
        self.mask = mask

        self.n_summary = s.n_summary
        assert len(mask) == s.n_summary, "Summary statistics mask has the wrong size"
        
        self.obs = obs

    def calc(self, repetition_list):
        ret = self.s.calc(repetition_list)
        mask = np.broadcast_to(self.mask, ret.shape)
        obs_vals = np.broadcast_to(self.obs_vals, ret.shape)
        ret[mask] = obs_vals[mask]
        return ret
