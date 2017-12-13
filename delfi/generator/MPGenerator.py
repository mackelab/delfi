import abc
import numpy as np

from delfi.utils.meta import ABCMetaDoc
from delfi.utils.progress import no_tqdm, progressbar

from delfi.generator.Default import Default


import multiprocessing as mp

class Worker(mp.Process):
    def __init__(self, conn, model, summary, seed=None):
        super().__init__()
        self.conn = conn
        self.model = model
        self.summary = summary
        self.rng = np.random.RandomState(seed=seed)

    def run(self):
        self.log("Starting worker")
        while True:
            try:
                self.log("Listening")
                params_batch = self.conn.recv()
            except EOFError:
                self.log("Leaving")
                break
            
            if len(params_batch) == 0:
                self.log("Skipping")
                self.conn.send(([], []))
                continue

            # run forward model for all params, each n_reps times
            self.log("Received data of size {}".format(len(params_batch)))
            self.log("Data: {}".format(params_batch))
            result = self.model.gen(params_batch)

            stats, params = self.process_batch(params_batch, result)

            self.log("Sending data")
            self.conn.send((stats, params))
            self.log("Done")

    def process_batch(self, params_batch, result):
        ret_stats = []
        ret_params = []

        # for every datum in data, check validity
        params_data_valid = []  # list of params with valid data
        data_valid = []  # list of lists containing n_reps dicts with data

        for param, datum in zip(params_batch, result):
            data_valid.append(datum)
            params_data_valid.append(param)

        # for every data in data, calculate summary stats
        for param, datum in zip(params_data_valid, data_valid):
            # calculate summary statistics
            sum_stats = self.summary.calc(datum)  # n_reps x dim stats

            # check validity
            ret_stats.append(sum_stats)
            # if sum stats is accepted, accept the param as well
            ret_params.append(param)

        return ret_stats, ret_params
    
    def log(self, msg):
        print("Worker: {}".format(msg))

class MPGenerator(Default):
    def __init__(self, models, prior, summary, seed=None):
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
        super().__init__(model=None, prior=prior, summary=summary, seed=None)
        pipes = [ mp.Pipe(duplex=True) for m in models ]
        self.workers = [ Worker(p[1], m, summary, seed=self.rng.randint(low=0,high=2**31)) for p, m in zip(pipes, models) ]
        self.pipes = [ p[0] for p in pipes ]

        self.log("Starting workers")
        for w in self.workers:
            w.start()

        self.log("Done")

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
        minibatches = self.iterate_minibatches(params, minibatch)
        done = False
        with pbar:
            while not done:
                for w, p in zip(self.workers, self.pipes):
                    try:
                        params_batch = next(minibatches)
                    except StopIteration:
                        done = True
                        break

                    self.log("Dispatching to worker (len = {})".format(len(params_batch)))
                    p.send(params_batch)
                    self.log("Done")

                for w, p in zip(self.workers, self.pipes):
                    self.log("Listening to worker")
                    stats, params = p.recv()
                    self.log("Done")
                    pbar.update(len(params))

                    final_stats += stats
                    final_params += params

        # TODO: for n_reps > 1 duplicate params; reshape stats array

        # n_samples x n_reps x dim theta
        params = np.array(final_params)

        # n_samples x n_reps x dim summary stats
        stats = np.array(final_stats)
        stats = stats.squeeze(axis=1)

        return params, stats

    def log(self, msg):
        print("Parent: {}".format(msg))
    def __del__(self):
        self.log("Closing")
        for w, p in zip(self.workers, self.pipes):
            self.log("Closing pipe")
            p.close()

        for w in self.workers:
            self.log("Joining process")
            w.join(timeout=1)
            w.terminate()
