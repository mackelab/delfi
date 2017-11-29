import delfi.distribution as dd
import delfi.generator as dg
import delfi.inference as infer
import delfi.summarystats as ds
import numpy as np

from delfi.simulator.Gauss import Gauss


def test_basic_inference(n_params=2, seed=42):
    m = Gauss(dim=n_params, seed=seed)
    p = dd.Gaussian(m=np.zeros((n_params, )), S=np.eye(n_params), seed=seed)
    s = ds.Identity()
    g = dg.Default(model=m, prior=p, summary=s)

    # set up inference
    res = infer.Basic(g, seed=seed)

    # run with N samples
    out = res.run(1000)

    # check result
    posterior = res.predict(np.array([0., 0.]).reshape(1, -1))
    assert np.allclose(posterior.xs[0].S, np.array([[0.1, 0.0],
                                                    [0.0, 0.1]]), atol=0.05)
    assert np.allclose(posterior.xs[0].m, np.array([0.0, 0.0]), atol=0.05)


def test_snpe_inference(n_params=2, seed=42):
    m = Gauss(dim=n_params, seed=seed)
    p = dd.Gaussian(m=np.zeros((n_params, )), S=np.eye(n_params), seed=seed)
    s = ds.Identity()
    g = dg.Default(model=m, prior=p, summary=s)

    # observation
    _, obs = g.gen(1)

    # set up inference
    res = infer.SNPE(g, obs=obs)

    # run with N samples
    out = res.run(n_train=1000, n_rounds=1)

    # check result
    posterior = res.predict(np.array([0., 0.]).reshape(1, -1))
    assert np.allclose(posterior.xs[0].S, np.array([[0.1, 0.0],
                                                    [0.0, 0.1]]), atol=0.05)
    assert np.allclose(posterior.xs[0].m, np.array([0.0, 0.0]), atol=0.05)
