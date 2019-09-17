import os
import uuid
import pickle
import subprocess
from delfi.generator.Default import Default


def run_remote(simulator_class,
               prior,  # might still be relevant with proposals due to rejection
               summary,
               n_samples,
               hostname,
               username,
               simulator_args=None,
               simulator_kwargs=None,
               remote_python_executable=None,
               remote_work_path=None,
               local_work_path=None,
               proposal=None,  # use prior by default
               n_workers=None,  # use number of remote cpus by default
               generator_seed=None,
               use_slurm=False,  # use the job manager with sbatch
               slurm_options=None,
               **generator_kwargs):
    """
    Create a MPGenerator on a remote server and generate samples.

    :param remote_python_executable:
    :param slurm_options:
    :param simulator_class:
    :param prior:
    :param summary:
    :param n_samples:
    :param hostname:
    :param username:
    :param simulator_args:
    :param simulator_kwargs:
    :param remote_python_executable:
    :param remote_work_path:
    :param local_work_path:
    :param proposal:
    :param n_workers:
    :param generator_seed:
    :param use_slurm:
    :param generator_kwargs:
    :return: (params, stats)
    """
    if simulator_args is None:
        simulator_args = []
    if simulator_kwargs is None:
        simulator_kwargs = dict()
    if remote_python_executable is None:
        remote_python_executable = 'python3'
    if remote_work_path is None:
        remote_work_path = './'
    if local_work_path is None:
        local_work_path = os.getenv('HOME')
    if local_work_path is None:
        local_work_path = os.getcwd()
    if slurm_options is None:
        slurm_options = dict()

    # define file names. important to use different local/remote names in case
    # we're ssh-ing into localhost etc.
    uu = str(uuid.uuid1())
    datafile_local = os.path.join(local_work_path,
                                  'local_data_{0}.pickle'.format(uu))
    datafile_remote = os.path.join(remote_work_path,
                                   'remote_data_{0}.pickle'.format(uu))
    samplefile_local = os.path.join(local_work_path,
                                    'local_samples_{0}.pickle'.format(uu))
    samplefile_remote = os.path.join(remote_work_path,
                                     'remote_samples_{0}.pickle'.format(uu))

    data = dict(simulator_class=simulator_class, simulator_args=simulator_args,
                simulator_kwargs=simulator_kwargs, prior=prior, summary=summary,
                proposal=proposal, n_samples=n_samples,
                generator_seed=generator_seed,
                n_workers=n_workers, generator_kwargs=generator_kwargs,
                samplefile=samplefile_remote, use_slurm=use_slurm,
                python_executable=remote_python_executable,
                slurm_options=slurm_options)

    with open(datafile_local, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # copy the data file to the remote host
    result = subprocess.run(
        ['scp', datafile_local,
         '{0}@{1}:{2}'.format(username, hostname, datafile_remote)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, \
        "failed to copy data file to remote host: {0}" \
        .format(result.stderr.decode())

    # generate samples remotely
    python_commands = 'from delfi.generator.MPGenerator import ' \
        'mpgen_from_file; mpgen_from_file(\'{0}\')'.format(datafile_remote)

    remote_command = '{0} -c \"{1}\"'.format(remote_python_executable,
                                             python_commands)

    result = subprocess.run(['ssh', hostname, '-l', username, remote_command],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    assert result.returncode == 0, \
        "failed to run code on remote host: {0}".format(result.stderr.decode())

    # copy samples back to local host
    result = subprocess.run(
        ['scp', '{0}@{1}:{2}'.format(username, hostname, samplefile_remote),
         samplefile_local],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, \
        "failed to copy samples file from remote host: {0}" \
        .format(result.stderr.decode())

    with open(samplefile_local, 'rb') as f:
        samples = pickle.load(f)

    # clean up by deleting data/sample files on remote/local machines
    os.remove(datafile_local)
    os.remove(samplefile_local)
    result = subprocess.run(['ssh', hostname, '-l', username,
                             'rm {0} && rm {1}'.format(datafile_remote,
                                                       samplefile_remote)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, \
        "failed to delete file(s) from remote host: {0}" \
        .format(result.stderr.decode())

    return samples['params'], samples['stats']


class RemoteGenerator(Default):
    def __init__(self,
                 simulator_class, prior, summary,
                 hostname, username,
                 simulator_args=None, simulator_kwargs=None,
                 remote_python_executable=None, use_slurm=False, slurm_options=None,
                 local_work_path=None, remote_work_path=None,
                 seed=None):
        """
        Generator that creates an MPGenerator on a remote server and uses that
        to run simulations.

        :param simulator_class:
        :param prior:
        :param summary:
        :param hostname:
        :param username:
        :param remote_python_executable:
        :param use_slurm:
        :param local_work_path:
        :param remote_work_path:
        :param seed:
        """
        super().__init__(model=None, prior=prior, summary=summary, seed=seed)
        self.simulator_class, self.hostname, self.username,\
            self.simulator_args, self.simulator_kwargs, \
            self.remote_python_executable, self.local_work_path,\
            self.remote_work_path, self.use_slurm, self.slurm_options = \
            simulator_class, hostname, username, simulator_args,\
            simulator_kwargs, remote_python_executable, local_work_path, \
            remote_work_path, use_slurm, slurm_options

    def gen(self, n_samples, n_workers=None, **kwargs):
        self.prior.reseed(self.gen_newseed())
        self.summary.reseed(self.gen_newseed())

        return \
            run_remote(self.simulator_class,
                       self.prior,
                       self.summary,
                       n_samples,
                       hostname=self.hostname,
                       username=self.username,
                       simulator_args=self.simulator_args,
                       simulator_kwargs=self.simulator_kwargs,
                       remote_python_executable=self.remote_python_executable,
                       remote_work_path=self.remote_work_path,
                       local_work_path=self.local_work_path,
                       proposal=self.proposal,
                       n_workers=n_workers,
                       generator_seed=self.gen_newseed(),
                       use_slurm=self.use_slurm,
                       slurm_options=self.slurm_options,
                       **kwargs)
