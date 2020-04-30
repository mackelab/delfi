import multiprocessing as mp
import numpy as np
import os
import sys
import subprocess
import pickle
from delfi.generator.Default import Default
from delfi.utils.progress import no_tqdm, progressbar
import time


class Worker(mp.Process):
    def __init__(self, n, queue, conn, model, summary, seed=None, verbose=False):
        super().__init__()
        self.n = n
        self.queue = queue
        self.verbose = verbose
        self.conn = conn
        self.model = model
        self.summary = summary
        self.rng = np.random.RandomState(seed=seed)

    def update(self, i):
        self.queue.put(i)

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
            result = self.model.gen(params_batch, pbar=self)

            stats, params = self.process_batch(params_batch, result)

            self.log("Sending data")
            self.queue.put((stats, params))
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

            ret_stats.append(sum_stats)
            ret_params.append(param)

        return ret_stats, ret_params

    def log(self, msg):
        if self.verbose:
            print("Worker {}: {}".format(self.n, msg))


def default_MPGenerator_rej(x):
    return 1


class MPGenerator(Default):
    def __init__(self, models, prior, summary, rej=None, seed=None, verbose=False):
        """Generator supporting multiprocessing

        Parameters
        ----------
        models : List of simulator instances
            Forward models
        prior : Distribution or Mixture instance
            Prior over parameters
        summary : SummaryStats instance
            Summary statistics
        rej : Function
            Rejection kernel

        Attributes
        ----------
        proposal : None or Distribution or Mixture instance
            Proposal prior over parameters. If specified, will generate
            samples given parameters drawn from proposal distribution rather
            than samples drawn from prior when `gen` is called.
        """
        super().__init__(model=None, prior=prior, summary=summary, seed=seed)

        self.rej = rej if rej is not None else default_MPGenerator_rej
        self.verbose = verbose
        self.models = models

        self.workers = None
        self.pipes = None
        self.queue = None

    def reseed(self, seed):
        """Carries out the following operations, in order:
        1) Reseeds the master RNG for the generator object, using the input seed
        2) Reseeds the prior from the master RNG. This may cause additional
        distributions to be reseeded using the prior's RNG (e.g. if the prior is
        a mixture)
        3) Reseeds the simulator RNG, from the master RNG
        4) Reseeds the proposal, if present
        """
        self.rng.seed(seed=seed)
        self.seed = seed
        self.prior.reseed(self.gen_newseed())
        for m in self.models:
            m.reseed(self.gen_newseed())
        if self.proposal is not None:
            self.proposal.reseed(self.gen_newseed())

    def start_workers(self):
        pipes = [ mp.Pipe(duplex=True) for m in self.models ]
        self.queue = mp.Queue()
        self.workers = [ Worker(i, self.queue, pipes[i][1], self.models[i], self.summary, seed=self.rng.randint(low=0,high=2**31), verbose=self.verbose) for i in range(len(self.models)) ]
        self.pipes = [ p[0] for p in pipes ]

        self.log("Starting workers")
        for w in self.workers:
            w.start()

        self.log("Done")

    def stop_workers(self):
        if not hasattr(self, "workers") or self.workers is None:
            return

        self.log("Closing")
        for w, p in zip(self.workers, self.pipes):
            self.log("Closing pipe")
            p.close()

        for w in self.workers:
            self.log("Joining process")
            w.join(timeout=1)
            w.terminate()

        self.queue.close()

        self.workers = None
        self.pipes = None
        self.queue = None

    def iterate_minibatches(self, params, minibatch=50):
        n_samples = len(params)

        for i in range(0, n_samples - minibatch + 1, minibatch):
            yield params[i:i + minibatch]

        rem_i = n_samples - (n_samples % minibatch)
        if rem_i != n_samples:
            yield params[rem_i:]

    def gen(self, n_samples, n_reps=1, skip_feedback=False, prior_mixin=0, verbose=True, **kwargs):
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
                                  verbose=verbose)

        return self.run_model(params, skip_feedback=skip_feedback, verbose=verbose, **kwargs)

    def run_model(self, params, minibatch=50, skip_feedback=False, keep_data=True, verbose=False):
        # Run forward model for params (in batches)
        if not verbose:
            pbar = no_tqdm()
        else:
            pbar = progressbar(total=len(params))
            desc = 'Run simulations '
            if type(verbose) == str:
                desc += verbose
            pbar.set_description(desc)

        self.start_workers()
        final_params = []
        final_stats = []  # list of summary stats
        minibatches = self.iterate_minibatches(params, minibatch)
        done = False
        with pbar:
            while not done:
                active_list = []
                for w, p in zip(self.workers, self.pipes):
                    try:
                        params_batch = next(minibatches)
                    except StopIteration:
                        done = True
                        break

                    active_list.append((w, p))
                    self.log("Dispatching to worker (len = {})".format(len(params_batch)))
                    p.send(params_batch)
                    self.log("Done")

                n_remaining = len(active_list)
                while n_remaining > 0:
                    self.log("Listening to worker")
                    msg = self.queue.get()
                    if type(msg) == int:
                        self.log("Received int")
                        pbar.update(msg)
                    elif type(msg) == tuple:
                        self.log("Received results")
                        stats, params = self.filter_data(*msg, skip_feedback=skip_feedback)
                        final_stats += stats
                        final_params += params
                        n_remaining -= 1
                    else:
                        self.log("Warning: Received unknown message of type {}".format(type(msg)))

        self.stop_workers()

        # TODO: for n_reps > 1 duplicate params; reshape stats array

        # n_samples x n_reps x dim theta
        params = np.array(final_params)

        # n_samples x n_reps x dim summary stats
        stats = np.array(final_stats)
        stats = stats.squeeze(axis=1)

        return params, stats

    def filter_data(self, stats, params, skip_feedback=False):
        if skip_feedback == True:
            return stats, params

        ret_stats = []
        ret_params = []

        for stat, param in zip(stats, params):
            response = self._feedback_summary_stats(stat)
            if response == 'accept':
                ret_stats.append(stat)
                ret_params.append(param)
            elif response == 'discard':
                continue
            else:
                raise ValueError('response not supported')

        return ret_stats, ret_params

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
        if self.rej(sum_stats):
            return 'accept'
        else:
            return 'discard'

    def log(self, msg):
        if self.verbose:
            print("Parent: {}".format(msg))

    def __del__(self):
        self.stop_workers()


def default_slurm_options():  # pragma: no cover
    opts = {'clusters': None,
            'time': '1:00:00',
            'D': os.path.expanduser('~'),
            'ntasks-per-node': 1,
            'nodes': 1,
            'output': os.path.join(os.path.expanduser('~'), '%j.out')
            }
    return opts


def generate_slurm_script(filename):  # pragma: no cover
    """
    Save a slurm script to run mpgen_from_file through a SLURM job manager

    :param filename:
    :return:
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    slurm_options = default_slurm_options()
    if data['slurm_options'] is not None:
        slurm_options.update(data['slurm_options'])
    assert slurm_options['clusters'] is not None, "cluster(s) must be specified"
    assert 'wait' not in slurm_options.keys() and 'W' not in slurm_options.keys(), "--wait/W always on, not options"

    slurm_script_file = os.path.splitext(filename)[0] + '_slurm.sh'
    with open(slurm_script_file, 'w') as f:

        f.write('#!/bin/bash\n')

        for key, val in slurm_options.items():
            if len(key) == 1:
                prefix = '-'
                postfix = ' '
            else:
                prefix = '--'
                postfix= '='
            s = '#SBATCH {0}{1}'.format(prefix, key)
            if val is not None:
                s += '{0}{1}'.format(postfix, val)
            f.write(s + '\n')

        f.write('#SBATCH --wait\n')  # block execution until the job finishes
        # f.write('source /etc/profile.d/modules.sh\n')  # for LRZ, may not be universal

        python_commands = 'from delfi.generator.MPGenerator import mpgen_from_file;'\
            'mpgen_from_file(\'{0}\', from_slurm=True)'.format(filename)
        f.write('srun {0} -c "{1}"\n'.format(data['python_executable'], python_commands))

    return slurm_options, slurm_script_file


def get_slurm_task_index():  # pragma: no cover
    localid = int(os.getenv('SLURM_LOCALID'))
    return int(os.getenv('SLURM_GTIDS').split(',')[localid])


def mpgen_from_file(filename, n_workers=None, from_slurm=False, cleanup=True):  # pragma: no cover
    """
    Run simulations from a file using a multi-process generator, and save them in a file.

    This function can be used as a stand-alone utility, but is mainly meant to be called on a remote host over ssh by a
    RemoteGenerator.

    :param cleanup:
    :param from_slurm:
    :param n_workers:
    :param filename: file describing simulations to be run
    :return:
    """
    start_time = time.time()
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    if from_slurm:  # this function is running on a slurm node

        tid = get_slurm_task_index()
        ncpus = os.getenv('SLURM_JOB_CPUS_PER_NODE')
        print('started task {0}, {1} cpus\n'.format(tid, ncpus))
        generator_seed = data['generator_seed'] + tid
        ntasks = int(os.getenv('SLURM_NTASKS'))

        sf, se = os.path.splitext(data['samplefile'])
        samplefile = sf + '_{0}'.format(tid) + se

        samples_per_task = int(np.ceil(data['n_samples'] / ntasks))
        n_samples = np.minimum((tid + 1) * samples_per_task, data['n_samples']) - tid * samples_per_task

    elif data['use_slurm']:  # start a slurm job that will call this function once per task

        slurm_options, slurm_script_file = generate_slurm_script(filename)
        ntasks = int(slurm_options['ntasks-per-node']) * int(slurm_options['nodes'])

        result = subprocess.run(['sbatch', slurm_script_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # sbatch will now block until job is completed due to --wait flag
        prefix = 'Submitted batch job '
        L = [s for s in result.stdout.decode().split('\n') if s.startswith(prefix)]
        assert len(L) == 1, "job was not submitted correctly"
        jobid = int(L[0][len(prefix):].split(' ')[0])
        if result.returncode != 0:  # e.g. job timed out
            sys.stderr.write('SLURM job {0} terminated abnormally: {1}'.format(jobid, result.stderr.decode()))

        # collect results from each task's file
        params, stats, task_time = None, None, np.full(ntasks, np.nan)
        for tid in range(ntasks):
            sf, se = os.path.splitext(data['samplefile'])
            samplefile_this_task = sf + '_{0}'.format(tid) + se
            if not os.path.exists(samplefile_this_task):
                continue

            with open(samplefile_this_task, 'rb') as f:
                samples = pickle.load(f)
                if params is None:
                    params, stats = samples['params'], samples['stats']
                else:
                    params, stats = np.vstack((params, samples['params'])), np.vstack((stats, samples['stats']))
                task_time[tid] = samples['time']
        assert params is not None, "failed to generate any samples"

        elapsed_time = time.time() - start_time

        # save all samples in one file
        with open(data['samplefile'], 'wb') as f:
            pickle.dump(dict(params=params, stats=stats, time=elapsed_time, task_time=task_time), f,
                        protocol=pickle.HIGHEST_PROTOCOL)

        if not cleanup:
            return

        #clean up all created files except the final samplefile:
        for tid in range(ntasks):
            sf, se = os.path.splitext(data['samplefile'])
            samplefile_this_task = sf + '_{0}'.format(tid) + se
            if not os.path.exists(samplefile_this_task):
                continue
            os.remove(samplefile_this_task)

        outputfile = slurm_options['output'].replace('%j', str(jobid))
        if '%' in outputfile:
            sys.stderr.write('output file(s) not be removed, only %t is supported')
        else:
            os.remove(outputfile)

        os.remove(slurm_script_file)

        return

    else:  # non-SLURM: use a single generator for all the samples

        n_samples, generator_seed, samplefile = data['n_samples'], data['generator_seed'], data['samplefile']

    if n_workers is None:
        n_workers = data['n_workers']
    if n_workers is None:
        n_workers = mp.cpu_count()
    n_workers = np.minimum(n_workers, n_samples)

    rng = np.random.RandomState(seed=generator_seed + 2500)
    summary_seed = rng.randint(0, 2 ** 31)
    summary = data['summary_class'](*data['summary_args'], seed=summary_seed, **data['summary_kwargs'])
    prior = data['prior']
    prior.reseed(rng.randint(0, 2 ** 31))

    if n_workers > 1:
        simulator_seeds = [rng.randint(0, 2 ** 31) for _ in range(n_workers)]

        models = [data['simulator_class'](*data['simulator_args'], seed=s, **data['simulator_kwargs'])
                  for s in simulator_seeds]
        g = MPGenerator(models, data['prior'], summary, seed=rng.randint(0, 2**31), verbose=False)
    else:
        s = rng.randint(0, 2 ** 31)
        model = data['simulator_class'](*data['simulator_args'], seed=s, **data['simulator_kwargs'])
        g = Default(model, data['prior'], summary, seed=rng.randint(0, 2**31))

    g.proposal = data['proposal']

    samples_remaining, params, stats = n_samples, None, None
    while samples_remaining > 0:

        if data['save_every'] is not None:
            next_batchsize = np.minimum(samples_remaining, data['save_every'])
        else:
            next_batchsize = samples_remaining

        next_params, next_stats = g.gen(next_batchsize, **data['generator_kwargs'])
        if params is None:
            params, stats = next_params, next_stats
        else:
            params, stats = np.vstack((params, next_params)), np.vstack((stats, next_stats))
        samples_remaining -= next_params.shape[0]

        elapsed_time = time.time() - start_time
        with open(samplefile, 'wb') as f:
            pickle.dump(dict(params=params, stats=stats, time=elapsed_time), f, protocol=pickle.HIGHEST_PROTOCOL)
