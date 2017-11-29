import delfi.kernel as dk
import numpy as np


def test_uniform_1d():
    obs = np.array([[0.]])
    k = dk.Uniform(obs=obs)

    x = np.array([[0.5]])
    val = k.eval(x)

    assert np.array_equal(val, np.asarray([1.0]))
