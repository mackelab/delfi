import abc
import numpy as np

from delfi.summarystats import BaseSummaryStats

class MaskedSummaryStats(BaseSummaryStats)
    def __init__(self, s, mask, obs, seed=None):
        """ Summary stats with imputed values

        This is a wrapper around BaseSummaryStats which imputes
        fixed values for specified statistics.

        Parameters
        ----------
        s : BaseSummaryStats
            The original summary stats
        mask : 1d array 
            Boolean array determining the values to be imputed. False corresponds to imputed entries.
        obs : 1d array
            Array of summary statistics from which to impute the values
        seed : int or None
            See BaseSummaryStats
        """

        super().__init__(seed=seed)
        self.s = s
        self.mask = mask

        self.n_summary = np.count_nonzero(self.mask)
        assert len(mask) == s.n_summary, "Summary statistics mask has the wrong size"
        
        self.obs = obs

    def calc(self, repetition_list):
        ret = self.s.calc(repetition_list)
        mask = np.broadcast_to(self.mask, ret.shape)
        obs_vals = np.broadcast_to(self.obs_vals, ret.shape)
        ret[~mask] = obs_vals[~mask]
        return ret
