import delfi.distribution as dd
import numpy as np
from scipy.special import expit, logit
from delfi.utils.bijection import named_bijection

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


def test_poisson(): 
    N = 50000
    mu = 5.
    offset = 1. 
    dist = dd.Poisson(mu=mu, offset=offset)
    samples = dist.gen(N)
    logprobs = dist.eval(samples)
    
    assert samples.shape == (N, 1)
    assert logprobs.shape == (N,)
    assert np.allclose(np.mean(samples, axis=0), mu + offset, atol=0.1)
     

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


def test_IndependentJoint_eval():
    N = 1000
    B1 = [-1.0, 1.0]
    B2 = [-2.0, 2.0]
    u1 = dd.Uniform(B1[0], B1[1])
    u2 = dd.Uniform(B2[0], B2[1])
    dist = dd.IndependentJoint([u1, u2])
    samples = dist.gen(N)
    logprobs = dist.eval(samples, log=True)
    lpdfval = -np.log((B2[1] - B2[0]) * (B1[1] - B1[0]))
    assert np.isclose(logprobs, lpdfval).all()

    assert samples.shape == (N, 2)
    assert logprobs.shape == (N,)


def test_IndependentJoint_marginals():
    N = 1000
    m = np.array([1., 3., 0.])
    S = [[8., 2., 1.],
         [2., 3., 2.],
         [1., 2., 3.]]
    gs = [dd.Gaussian(m=m + i, S=S) for i in [-1, 0, 1]]
    dist = dd.IndependentJoint(gs)
    samples = dist.gen(N)
    log_probs = dist.eval(samples, log=True)
    jjs = [np.arange(3 * i, 3 * (i + 1)) for i in range(3)]
    log_marginals = [dist.eval(samples[:, jj], ii = jj) for jj in jjs]
    assert np.isclose(log_probs, np.vstack(log_marginals).sum(axis=0)).all()

    log_submarginal_1 = dist.dists[0].eval(samples[:, [1]], ii=[1])
    log_submarginal_4 = dist.dists[1].eval(samples[:, [4]], ii=[1])
    log_submarginal_6_7_8 = dist.dists[2].eval(samples[:, [6, 7, 8]])
    log_submarginal_1_4_6_7_8 = dist.eval(samples[:, [1, 4, 6, 7, 8]],
                                    ii=[1, 4, 6, 7, 8])
    assert np.isclose(log_submarginal_1_4_6_7_8,
                      log_submarginal_1 + log_submarginal_4 +
                      log_submarginal_6_7_8).all()


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


def test_TransformedDistribution(seed=5, nsamples=1000, ndim=2):
    lower = np.random.rand(ndim)
    upper = np.random.rand(ndim) + 2
    dist = dd.Uniform(lower, upper)

    dist.reseed(seed)
    z = dist.gen(nsamples)

    f, finv, f_jac_logD, finv_jac_logD = \
        named_bijection('affine',
                        scale=1.0 / (upper - lower),
                        offset=-lower / (upper - lower))

    g, ginv, g_jac_logD, ginv_jac_logD = named_bijection('logit')

    bijection = lambda x: g(f(x))
    inverse_bijection = lambda y: finv(ginv(y))
    bijection_jac_logD = lambda x: g_jac_logD(f(x)) + f_jac_logD(x)

    #inputscale = lambda x: (x - lower) / (upper - lower)
    #bijection = lambda x: logit(inputscale(x))  # logit function with scaled input
    #inverse_bijection = lambda y: expit(y) * (upper - lower) + lower  # logistic function with scaled output
    #bijection_jac_logD = lambda x: -(np.log(inputscale(x) * (1 - inputscale(x))) + np.log(upper - lower)).sum(axis=-1)

    dist_transformed = dd.TransformedDistribution(distribution=dist,
                                                  bijection=bijection,
                                                  inverse_bijection=inverse_bijection,
                                                  bijection_jac_logD=bijection_jac_logD)
    dist_transformed.reseed(seed)
    z_transformed = dist_transformed.gen(nsamples)
    assert np.allclose(z_transformed, bijection(z), atol=1e-8)

    dist_logistic = dd.Logistic(mu=np.zeros(ndim))

    assert np.allclose(dist_logistic.eval(z_transformed, log=False), dist_transformed.eval(z_transformed, log=False)), \
        "incorrect density"
    assert np.allclose(dist_logistic.eval(z_transformed, log=True), dist_transformed.eval(z_transformed, log=True)), \
        "incorrect log density"

