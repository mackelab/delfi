import delfi.summarystats as dss
import numpy as np


def test_mean_1_sample_1_feature():
    obs = [{'data': np.array([-1., 1.])},
           {'data': np.array([1., 2.])}]

    s = dss.Mean()
    result = s.calc(obs)

    expected = np.asarray([np.mean(o['data'])
                           for o in obs]).reshape(len(obs), 1)

    assert np.array_equal(result, expected)


def test_identity_1_sample_1_feature():
    obs = [{'data': np.array([-1., 1.])},
           {'data': np.array([1., 2.])}]

    s = dss.Identity()
    result = s.calc(obs)

    expected = np.asarray([o['data'] for o in obs])

    assert np.array_equal(result, expected)
