import getpass
import sys
import os
import subprocess
import delfi.distribution as dd
import delfi.generator as dg
import delfi.summarystats as ds
import numpy as np
from delfi.simulator.Gauss import Gauss


def test_gauss_shape():
    for n_params in range(1, 3):
        m = Gauss(dim=n_params)
        p = dd.Gaussian(m=np.zeros((n_params, )), S=np.eye(n_params))
        s = ds.Identity()

        g = dg.Default(model=m, prior=p, summary=s)

        n_samples = 100
        params, stats = g.gen(n_samples)

        n_summary = n_params
        assert params.shape == (n_samples, n_params)
        assert stats.shape == (n_samples, n_summary)


def test_IndependentJoint_uniform_rejection():
    # check that proposed samples are correctly rejected when using a
    # IndependentJoint prior with some child distributions uniform. We used a
    # Gaussian proposal to generate some samples that need to be rejected.
    N = 1000
    B1 = [-1.0, 1.0]
    B2 = [-2.0, 2.0]
    u1 = dd.Uniform(B1[0], B1[1])
    u2 = dd.Uniform(B2[0], B2[1])
    prior = dd.IndependentJoint([u1, u2])

    m = [0., 0.]
    S = [[2., 0.,],
         [0., 2.,]]
    proposal = dd.Gaussian(m=m, S=S)

    model = Gauss(dim=2)

    s = ds.Identity()

    g = dg.Default(model=model, prior=prior, summary=s)
    g.proposal = proposal

    params, stats = g.gen(N, verbose=False)
    assert (params.min(axis=0) >= np.array([B1[0], B2[0]])).all() and \
        (params.min(axis=0) <= np.array([B1[1], B2[1]])).all(), \
        "rejection failed"


def test_mpgen(n_samples=1000, n_params=2, n_cores=4, seed=500):
    p = dd.Gaussian(m=np.zeros((n_params,)), S=np.eye(n_params), seed=seed)
    s = ds.Identity(seed=seed + 1)

    mlist = [Gauss(dim=n_params, seed=seed + 2 + i) for i in range(n_cores)]
    g = dg.MPGenerator(models=mlist, prior=p, summary=s,
                       seed=seed + 2 + n_cores)
    params, stats = g.gen(n_samples, verbose=False)

    # make sure the different models are providing different outputs
    assert np.unique(params.size) == params.size
    assert np.unique(stats.size) == stats.size


def test_remotegen(n_samples=1000, n_params=2, seed=66, run_diagnostics=False):
    """
    test the RemoteGenerator by using the local machine to ssh into itself.
    For this test to succeed, an ssh private key will need to be added to the
    ssh agent, and the corresponding public key added to authorized_keys

    NOTE: This test will fail on travis unless eval $(ssh-agent -s) is run in the main script. Running it here using
    os.system seems to have no effect.
    """
    p = dd.Gaussian(m=np.zeros((n_params,)), S=np.eye(n_params), seed=seed)

    simulator_kwargs = dict(dim=2)

    hostname = '127.0.0.1'
    username = getpass.getuser()

    # in a real-world scenario, we would have already manually authenticated
    # the host. what we're doing here is a big security risk, but for localhost
    # it's (probably?) ok
    os.system('cp ~/.ssh/known_hosts ~/.ssh/known_hosts_backup')
    os.system('cp ~/.ssh/authorized_keys ~/.ssh/authorized_keys_backup')
    os.system('ssh-keyscan -H {0} >> ~/.ssh/known_hosts'.format(hostname))
    # generate a key-pair to use on localhost
    os.system('ssh-keygen -b 2048 -t rsa -f ~/.ssh/test_remotegen -q -N ""')
    os.system('ssh-add ~/.ssh/test_remotegen')   # add private key for client side
    os.system('cat ~/.ssh/test_remotegen.pub >> ~/.ssh/authorized_keys')

    if run_diagnostics:
        # run some diagnostics and print results to stderr
        sshdir = os.path.expanduser('~/.ssh/')
        sys.stderr.write(subprocess.run(['ssh-add', '-l'], stdout=subprocess.PIPE).stdout.decode() + '\n\n')
        sys.stderr.write(subprocess.run(['ls', sshdir], stdout=subprocess.PIPE).stdout.decode() + '\n\n')
        sys.stderr.write(subprocess.run(['cat', os.path.join(sshdir, 'authorized_keys')],
                                        stdout=subprocess.PIPE).stdout.decode() + '\n\n')
        sys.stderr.write(subprocess.run(['cat', os.path.join(sshdir, 'known_hosts')],
                                        stdout=subprocess.PIPE).stdout.decode() + '\n\n')
        sys.stderr.write(subprocess.run(['ls', '-ld', sshdir], stdout=subprocess.PIPE).stdout.decode() + '\n\n')
        sys.stderr.write(subprocess.run(['ls', '-l', sshdir], stdout=subprocess.PIPE).stdout.decode() + '\n\n')

    try:
        g = dg.RemoteGenerator(simulator_class=Gauss,
                               prior=p, summary_class=ds.Identity,
                               hostname=hostname,
                               username=username,
                               simulator_kwargs=simulator_kwargs,
                               use_slurm=False,
                               remote_python_executable=sys.executable,
                               seed=seed+2)
        params, stats = g.gen(n_samples, verbose=False)
        success = True
    except Exception as e:
        success = False
        err = e

    # restore ssh to previous state etc.
    os.system('ssh-add -d ~/.ssh/test_remotegen')
    os.system('mv ~/.ssh/known_hosts_backup ~/.ssh/known_hosts')
    os.system('mv ~/.ssh/authorized_keys_backup ~/.ssh/authorized_keys')
    os.system('rm ~/.ssh/test_remotegen*')

    if not success:
        raise err

    # make sure the different models are providing different outputs
    assert np.unique(params.size) == params.size
    assert np.unique(stats.size) == stats.size


def dont_test_remotegen_slurm(n_samples=500, n_params=2, seed=66, save_every=200,
                             hostname=None, username=None, clusters=None,
                             remote_python_executable=None, remote_work_path=None):
    assert type(hostname) is str and type(username) is str, "hostname and username must be provided"
    assert type(clusters) is str, "cluster(s) must be specified as a (comma-delimited) string"
    '''this test is currently disabled because we don't have slurm running on travis right now. it works fine
    if it's run locally on a machine with key-based ssh-access to a SLURM cluster, as of 17.09.2019'''

    p = dd.Gaussian(m=np.zeros((n_params,)), S=np.eye(n_params), seed=seed)

    simulator_kwargs = dict(dim=2)

    slurm_options = {'clusters': clusters,
                     'time': '0:10:00',
                     'ntasks-per-node': 2,
                     'nodes': 2}

    g = dg.RemoteGenerator(simulator_class=Gauss,
                           prior=p, summary_class=ds.Identity,
                           hostname=hostname,
                           username=username,
                           simulator_kwargs=simulator_kwargs,
                           use_slurm=True,
                           remote_python_executable=remote_python_executable,
                           remote_work_path=remote_work_path,
                           slurm_options=slurm_options,
                           save_every=save_every,
                           seed=seed + 2)
    params, stats = g.gen(n_samples, verbose=False)
    return params, stats