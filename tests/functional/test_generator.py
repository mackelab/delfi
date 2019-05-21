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
