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
