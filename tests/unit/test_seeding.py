import numpy as np
from copy import deepcopy
import delfi.distribution as dd
import delfi.simulator as sims
from delfi.summarystats import Identity
import delfi.generator as gen
import delfi.inference as inf

"""
These tests verify that computations depending on an rng can be precisely
repeated by resetting the rng to a previous state
"""

def test_rng_repeatability():
    mu = np.atleast_1d([0.0])
    S = np.atleast_2d(1.0)

    # distributions
    pG = dd.Gaussian(m=mu, S=S)
    check_repeatability_dist(pG)

    pMoG = dd.MoG(a = np.array([0.25, 0.75]), ms=[mu, mu], Ss=[S, S])
    check_repeatability_dist(pMoG)

    # simulators
    mG = sims.Gauss()
    check_repeatability_sim(mG, np.zeros(mG.dim_param).reshape(-1, 1))

    mMoG = sims.GaussMixture()
    check_repeatability_sim(mMoG, np.zeros(mMoG.dim_param).reshape(-1, 1))

    # generators
    g = gen.Default(model=mMoG, prior=pMoG, summary=Identity())
    check_repeatability_gen(g)

    # inference methods
    # we're going to create each one with a different deepcopy of g to make
    # sure thre are are no side effects e.g. changes to the proposal
    x0 = g.gen(1, verbose=False)[1]
    inf_opts = dict(obs=x0, n_components=2, n_hiddens=[5, 5], verbose=False,
                    pilot_samples=0)

    yB_nosvi = inf.Basic(deepcopy(g), svi=False, **inf_opts)
    check_repeatability_infer(yB_nosvi)

    yB_svi = inf.Basic(deepcopy(g), svi=True, **inf_opts)
    check_repeatability_infer(yB_svi)

    # skip CDELFI for now since it might crash if we don't use the prior
    #yC = inf.CDELFI(deepcopy(g), **inf_opts)
    #check_repeatability_infer(yC)

    yS = inf.SNPEB(deepcopy(g), prior_mixin=0.5, **inf_opts)
    check_repeatability_infer(yS)


def check_repeatability_infer(y, seed=42, n_train=10):
    """
    Run an inference algorithm twice, and make sure all results are the same.
    """
    y.reset(seed=seed)
    log1, train_data1, posterior1 = y.run(n_train=n_train)
    a1 = y.predict(y.obs).gen(5)

    y.reset(seed=seed)
    log2, train_data2, posterior2 = y.run(n_train=n_train)
    a2 = y.predict(y.obs).gen(5)

    """For a single-round algorithm train_data will be returned as a tuple.
    For multiround methods, it's a list of tuples. log and posterior are
    similar."""
    if type(train_data1) is not list:
        train_data1, train_data2 = [train_data1], [train_data2]
        log1, log2 = [log1], [log2]
        posterior1, posterior2 = [posterior1], [posterior2]

    assert len(train_data1) == len(train_data2), "Different number of rounds"
    for r in range(len(train_data1)):  # for each round
        assert np.all(train_data1[r][0] == train_data2[r][0]), \
            "different parameters"
        assert np.all(train_data1[r][1] == train_data2[r][1]), \
            "different sufficient statistic values"
        assert np.all(posterior1[r].a == posterior2[r].a), \
            "different mixture coefficients"
        for x1, x2 in zip(posterior1[r].xs, posterior2[r].xs):
            assert np.all(x1.m == x2.m), "Component means do not match"
            assert np.all(x1.S == x2.S), "Component covariances do not match"
    assert np.all(a1==a2), "posterior samples do not match"


def check_repeatability_dist(p, n=100, seed=42):
    p.reseed(seed)
    params1 = p.gen(n)
    p.reseed(seed)
    params2 = p.gen(n)
    assert np.all(params1 == params2)


def check_repeatability_sim(m, params, n=100, seed=42):
    m.reseed(seed)
    x1 = m.gen(params, n_reps=n)
    m.reseed(seed)
    x2 = m.gen(params, n_reps=n)
    assert np.all(x1 == x2)


def check_repeatability_gen(g, n=100, seed=42):
    # use prior_mixin to test the RNG of the generator too.
    g.reseed(seed)
    params1, stats1 = g.gen(n, prior_mixin=0.5, verbose=False)
    g.reseed(seed)
    params2, stats2 = g.gen(n, prior_mixin=0.5, verbose=False)
    assert np.all(params1 == params2), "parameters do not match"
    assert np.all(stats1 == stats2), "stats do not match"
