import delfi.distribution as dd
import numpy as np

from delfi.generator.Default import Default


class RejKernel(Default):

    def __init__(self, rej, **kwargs):
        """Generator

        Parameters
        ----------
        model : Simulator instance
            Forward model
        prior : Distribution or Mixture instance
            Prior over parameters
        summary : SummaryStats instance
            Summary statistics
        rej : rejection function
            Rejection kernel implementation

        Attributes
        ----------
        proposal : None or Distribution or Mixture instance
            Proposal prior over parameters. If specified, will generate
            samples given parameters drawn from proposal distribution rather
            than samples drawn from prior when `gen` is called.
        """
        super().__init__(**kwargs)
        self.rej = rej

    @copy_ancestor_docstring
    def _feedback_summary_stats(self, sum_stats):
        if self.rej(sum_stats):
            return 'accept'
        else: 
            return 'discard'
