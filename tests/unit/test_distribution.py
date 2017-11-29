import delfi.distribution as dd
import numpy as np

seed = 42


def test_discrete_gen():
    N = 25000
    p = 0.9
    dist = dd.Discrete(np.array([1 - p, p]), seed=seed)
    samples = dist.gen(N)

    assert samples.shape == (N, 1)
    assert np.isclose(np.sum(samples) / N, p, atol=0.01)


def test_gaussian_1d():
    N = 50000
    m = [1.]
    S = [[3.]]
    dist = dd.Gaussian(m=m, S=S, seed=seed)
    samples = dist.gen(N)
    logprobs = dist.eval(samples)

    assert samples.shape == (N, 1)
    assert logprobs.shape == (N,)
    assert np.isclose(np.mean(samples).reshape(-1), m, atol=0.1)
    assert np.isclose(
        np.cov(samples, rowvar=False).reshape(-1, 1), S, atol=0.1)


def test_gaussian_3d():
    N = 50000
    m = [1., 3., 0.]
    S = [[8., 2., 1.],
         [2., 3., 2.],
         [1., 2., 3.]]
    dist = dd.Gaussian(m=m, S=S, seed=seed)
    samples = dist.gen(N)
    logprobs = dist.eval(samples)

    assert samples.shape == (N, 3)
    assert logprobs.shape == (N,)
    assert np.allclose(np.mean(samples, axis=0), m, atol=0.1)
    assert np.allclose(np.cov(samples, rowvar=False), S, atol=0.1)


def test_studentst_1d():
    N = 100000
    m = [1.]
    S = [[3.]]
    dof = 1000
    dist = dd.StudentsT(m=m, S=S, dof=dof, seed=seed)
    samples = dist.gen(N)
    logprobs = dist.eval(samples)

    St = np.asarray(S) * (dof / (dof - 2))
    assert samples.shape == (N, 1)
    assert logprobs.shape == (N,)
    assert np.isclose(np.mean(samples).reshape(-1), m, atol=0.1)
    assert np.isclose(
        np.cov(samples, rowvar=False).reshape(-1, 1), St, atol=0.1)


def test_studentst_3d():
    N = 100000
    m = [1., 3., 0.]
    S = [[8., 2., 1.],
         [2., 3., 2.],
         [1., 2., 3.]]
    dof = 1000
    dist = dd.StudentsT(m=m, S=S, dof=dof, seed=seed)
    samples = dist.gen(N)
    logprobs = dist.eval(samples)
    St = np.asarray(S) * (dof / (dof - 2))

    assert samples.shape == (N, 3)
    assert logprobs.shape == (N,)
    assert np.allclose(np.mean(samples, axis=0), m, atol=0.1)
    assert np.allclose(np.cov(samples, rowvar=False), St, atol=0.1)


def test_uniform_1d():
    N = 1000
    lower = [1.]
    upper = [2.]
    dist = dd.Uniform(lower, upper, seed=seed)
    samples = dist.gen(N)
    logprobs = dist.eval(samples)

    assert samples.shape == (N, 1)
    assert logprobs.shape == (N,)


def test_uniform_2d():
    N = 1000
    lower = [1., 3.]
    upper = [2., 4.]
    dist = dd.Uniform(lower, upper, seed=seed)
    samples = dist.gen(N)
    logprobs = dist.eval(samples)

    assert samples.shape == (N, 2)
    assert logprobs.shape == (N,)


def test_mixture_of_gaussians_1d():
    N = 1000
    m = [1.]
    S = [[3.]]
    ms = [m, m]
    Ss = [S, S]
    dist = dd.MoG(a=[0.5, 0.5], ms=ms, Ss=Ss, seed=seed)
    samples = dist.gen(N)
    logprobs = dist.eval(samples)

    assert samples.shape == (N, 1)
    assert logprobs.shape == (N,)


def test_mixture_of_gaussians_3d():
    N = 1000
    m = [1., 3., 0.]
    S = [[8., 2., 1.],
         [2., 3., 2.],
         [1., 2., 3.]]
    ms = [m, m]
    Ss = [S, S]
    dist = dd.MoG(a=[0.5, 0.5], ms=ms, Ss=Ss, seed=seed)
    samples = dist.gen(N)
    logprobs = dist.eval(samples)

    assert samples.shape == (N, 3)
    assert logprobs.shape == (N,)


def test_mixture_of_studentst_1d():
    N = 1000
    m = [1.]
    S = [[3.]]
    dof = 1000
    ms = [m, m]
    Ss = [S, S]
    dofs = [dof, dof]
    dist = dd.MoT(a=[0.5, 0.5], ms=ms, Ss=Ss, dofs=dofs, seed=seed)
    samples = dist.gen(N)
    logprobs = dist.eval(samples)

    assert samples.shape == (N, 1)
    assert logprobs.shape == (N,)


def test_mixture_of_studentst_3d():
    N = 1000
    m = [1., 3., 0.]
    S = [[8., 2., 1.],
         [2., 3., 2.],
         [1., 2., 3.]]
    dof = 1000
    ms = [m, m]
    Ss = [S, S]
    dofs = [dof, dof]
    dist = dd.MoT(a=[0.5, 0.5], ms=ms, Ss=Ss, dofs=dofs, seed=seed)
    samples = dist.gen(N)
    logprobs = dist.eval(samples)

    assert samples.shape == (N, 3)
    assert logprobs.shape == (N,)
