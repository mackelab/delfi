import abc
import numpy as np

from delfi.utils.meta import ABCMetaDoc
from delfi.utils.progress import no_tqdm, progressbar


class BaseGenerator(metaclass=ABCMetaDoc):
    def __init__(self, model, prior, summary, seed=None):
        """Generator

        Parameters
        ----------
        model : Simulator instance
            Forward model
        prior : Distribution or Mixture instance
            Prior over parameters
        summary : SummaryStats instance
            Summary statistics

        Attributes
        ----------
        proposal : None or Distribution or Mixture instance
            Proposal prior over parameters. If specified, will generate
            samples given parameters drawn from proposal distribution rather
            than samples drawn from prior when `gen` is called.
        """
        self.model = model
        self.prior = prior
        self.summary = summary
        self.proposal = None

        self.rng = np.random.RandomState(seed=seed)

    def draw_params(self, n_samples, skip_feedback=False, prior_mixin=0, verbose=True):
        if not verbose:
            pbar = no_tqdm()
        else:
            pbar = progressbar(total=n_samples)
            desc = 'Draw parameters '
            if type(verbose) == str:
                desc += verbose
            pbar.set_description(desc)

        # collect valid parameter vectors from the prior
        params = []  # list of parameter vectors
        with pbar:
            i = 0
            while i < n_samples:
                # sample parameter
                if self.proposal is None or self.rng.random_sample() < prior_mixin:
                    proposed_param = self.prior.gen(n_samples=1)  # dim params,
                else:
                    proposed_param = self.proposal.gen(n_samples=1)

                # check if parameter vector is valid
                response = self._feedback_proposed_param(proposed_param)
                if response == 'accept' or skip_feedback:
                    # add valid param vector to list
                    params.append(proposed_param.reshape(-1))
                    i += 1
                    pbar.update(1)
                elif response == 'resample':
                    # continue without increment on i or updating the bar
                    continue
                else:
                    raise ValueError('response not supported')

            return params

    def iterate_minibatches(self, params, minibatch=50):
        n_samples = len(params)

        for i in range(0, n_samples - minibatch+1, minibatch):
            yield params[i:i + minibatch]

        rem_i = n_samples - (n_samples % minibatch)
        if rem_i != n_samples:
            yield params[rem_i:]    

    def gen(self, n_samples, n_reps=1, skip_feedback=False, prior_mixin=0, minibatch=50, keep_data=True, verbose=True):
        """Draw parameters and run forward model

        Parameters
        ----------
        n_samples : int
            Number of samples
        n_reps: int
            Number of repetitions per parameter sample
        skip_feedback: bool
            If True, feedback checks on params, data and sum stats are skipped
        verbose : bool or str
            If False, will not display progress bars. If a string is passed,
            it will be appended to the description of the progress bar.

        Returns
        -------
        params : n_samples x n_reps x n_params
            Parameters
        stats : n_samples x n_reps x n_summary
            Summary statistics of data
        """
        assert n_reps == 1, 'n_reps > 1 is not yet supported'

        params = self.draw_params(n_samples=n_samples,
                                  skip_feedback=skip_feedback, 
                                  prior_mixin=prior_mixin,
                                  verbose = verbose)

        # Run forward model for params (in batches)
        if not verbose:
            pbar = no_tqdm()
        else:
            pbar = progressbar(total=len(params))
            desc = 'Run simulations '
            if type(verbose) == str:
                desc += verbose
            pbar.set_description(desc)

        final_params = []
        final_stats = []  # list of summary stats
        with pbar:
            for params_batch in self.iterate_minibatches(params, minibatch):
                # run forward model for all params, each n_reps times
                result = self.model.gen(params_batch, n_reps=n_reps, pbar=pbar)

                stats, params = self.process_batch(params_batch, result)
                final_params += params
                final_stats += stats

        # TODO: for n_reps > 1 duplicate params; reshape stats array

        # n_samples x n_reps x dim theta
        params = np.array(final_params)

        # n_samples x n_reps x dim summary stats
        stats = np.array(final_stats)
        stats = stats.squeeze(axis=1)

        return params, stats

    def process_batch(self, params_batch, result):
        ret_stats = []
        ret_params = []

        # for every datum in data, check validity
        params_data_valid = []  # list of params with valid data
        data_valid = []  # list of lists containing n_reps dicts with data

        for param, datum in zip(params_batch, result):
            # check validity
            response = self._feedback_forward_model(datum)
            if response == 'accept' or skip_feedback:
                data_valid.append(datum)
                # if data is accepted, accept the param as well
                params_data_valid.append(param)
            elif response == 'discard':
                continue
            else:
                raise ValueError('response not supported')

        # for every data in data, calculate summary stats
        for param, datum in zip(params_data_valid, data_valid):
            # calculate summary statistics
            sum_stats = self.summary.calc(datum)  # n_reps x dim stats

            # check validity
            response = self._feedback_summary_stats(sum_stats)
            if response == 'accept' or skip_feedback:
                ret_stats.append(sum_stats)
                # if sum stats is accepted, accept the param as well
                ret_params.append(param)
            elif response == 'discard':
                continue
            else:
                raise ValueError('response not supported')

        return ret_stats, ret_params

    @abc.abstractmethod
    def _feedback_proposed_param(self, param):
        """Feedback step after parameter has been proposed

        Parameters
        ----------
        param : np.array
            Parameter

        Returns
        -------
        response : str
            Supported responses are in ['accept', 'resample']
        """
        # TODO: check if parameter is inside of support of prior when
        # proposal distribution was used for sampling
        return 'accept'

    @abc.abstractmethod
    def _feedback_forward_model(self, data):
        """Feedback step after forward model ran

        Parameters
        ----------
        data : np.array
            Data

        Returns
        -------
        response : str
            Supported responses are in ['accept', 'discard']
        """
        return 'accept'

    @abc.abstractmethod
    def _feedback_summary_stats(self, sum_stats):
        """Feedback step after summary stats were computed

        Parameters
        ----------
        sum_stats : np.array
            Summary stats

        Returns
        -------
        response : str
            Supported responses are in ['accept', 'discard']
        """
        return 'accept'
