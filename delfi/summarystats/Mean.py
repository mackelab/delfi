import numpy as np

from delfi.summarystats.BaseSummaryStats import BaseSummaryStats


class Mean(BaseSummaryStats):
    """Reduces data to mean
    """

    def __init__(self, seed=None):
        super().__init__(seed=seed)
        # should return a matrix n_samples x 1 (mean)
        self.n_summary = 1

    @copy_ancestor_docstring
    def calc(self, repetition_list):
        # See BaseSummaryStats.py for docstring

        # get the number of repetitions contained
        n_reps = len(repetition_list)

        # build a matrix of n_reps x 1
        repetition_stats_matrix = np.zeros((n_reps, self.n_summary))

        # for every repetition, take the mean of the data in the dict
        for rep_idx, rep_dict in enumerate(repetition_list):
            repetition_stats_matrix[rep_idx, ] = np.mean(rep_dict['data'])

        return repetition_stats_matrix
