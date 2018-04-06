import delfi.distribution as dd
import numpy as np

from delfi.generator.BaseGenerator import BaseGenerator


class Default(BaseGenerator):
    @copy_ancestor_docstring
    def _feedback_proposed_param(self, param):
        # See BaseGenerator for docstring

        # if prior is uniform, reject samples outside of bounds
        # samples might be drawn outside bounds due to proposal
        if isinstance(self.prior, dd.Uniform):
            if np.any(param < self.prior.lower) or \
               np.any(param > self.prior.upper):
                return 'resample'
        elif isinstance(self.prior, dd.StackedDistribution):
            for p, ii in zip(self.prior.ps, self.prior.ndims):
                if isinstance(p, dd.Uniform):
                    if np.any(param[:,ii] < p.lower) or \
                       np.any(param[:,ii] > p.upper):
                        return 'resample' 

                elif isinstance(p, dd.Gamma):
                    if np.any(param[:,ii] < p.offset):
                        return 'resample'

        return 'accept'

    @copy_ancestor_docstring
    def _feedback_forward_model(self, data):
        # See BaseGenerator for docstring
        return 'accept'

    @copy_ancestor_docstring
    def _feedback_summary_stats(self, sum_stats):
        # See BaseGenerator for docstring
        return 'accept'
