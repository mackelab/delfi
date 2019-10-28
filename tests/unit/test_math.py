import numpy as np
import theano
import delfi.utils.math as mm
import delfi.utils.symbolic as sym
from delfi.distribution import Gaussian
from delfi.utils.bijection import named_bijection
from delfi.neuralnet.NeuralNet import dtype


def test_gaussprodZ():
    x1 = Gaussian(m=np.array([0.25, 0.5]), S=np.array([[0.6, 0.1], [0.1, 0.5]]))
    x2 = Gaussian(m=np.array([-0.1, 0.2]), S=np.array([[1.5, -0.1], [-0.1, 1.25]]))
    xprod = x1 * x2  # new Gaussian random variable from pdf product
    logZ = mm.gaussprodintegral(x1, x2, log=True)

    test_pts = np.vstack((x1.m, x2.m, (x1.m + x2.m) / 2.0, xprod.m, 0.0 * x1.m))
    lpdfprod = x1.eval(test_pts, log=True) + x2.eval(test_pts, log=True)
    lpdfprod_normed = xprod.eval(test_pts, log=True)

    assert np.isclose(lpdfprod_normed + logZ, lpdfprod).all(), "wrong Z (prod)"


def test_gaussquotientZ():
    x1 = Gaussian(m=np.array([0.25, 0.5]), S=np.array([[0.6, 0.1], [0.1, 0.5]]))
    x2 = Gaussian(m=np.array([-0.1, 0.2]), S=np.array([[1.5, -0.1], [-0.1, 1.25]]))
    xquot = x1 / x2  # new Gaussian random variable from pdf product
    logZ = mm.gaussquotientintegral(x1, x2, log=True)

    test_pts = np.vstack((x1.m, x2.m, (x1.m + x2.m) / 2.0, xquot.m, 0.0 * x1.m))

    lpdfquot = x1.eval(test_pts, log=True) - x2.eval(test_pts, log=True)
    lpdfquot_normed = xquot.eval(test_pts, log=True)
    assert np.isclose(lpdfquot_normed + logZ, lpdfquot).all(), "wrong Z (quot)"


def test_gaussprodquotientZ():
    x1 = Gaussian(m=np.array([0.25, 0.5]), S=np.array([[0.6, 0.1], [0.1, 0.5]]))
    x2 = Gaussian(m=np.array([-0.1, 0.2]), S=np.array([[1.5, -0.1], [-0.1, 1.25]]))
    x3 = Gaussian(m=np.array([-0.05, 0.3]), S=np.array([[3.5, -0.1], [-0.1, 4.25]]))
    xprodquot = (x1 * x2) / x3  # new Gaussian random variable from pdf product
    logZ = mm.gaussprodquotientintegral(x1, x2, x3, log=True)

    test_pts = np.vstack((x1.m, x2.m, x3.m, (x1.m + x2.m + x3.m) / 3.0,
                          xprodquot.m, 0.0 * x1.m))

    lpdfprodquot = x1.eval(test_pts, log=True) + x2.eval(test_pts, log=True) \
        - x3.eval(test_pts, log=True)
    lpdfprodquot_normed = xprodquot.eval(test_pts, log=True)
    assert np.isclose(lpdfprodquot_normed + logZ, lpdfprodquot).all(), "wrong Z (prod-quot)"


def test_bijections(dim=2, nsamples=1000, seed=1):
    rng = np.random.RandomState(seed=seed)

    for name in ['affine', 'logit', 'norminvcdf']:

        kwargs = dict()
        if name == 'affine':
            kwargs.update(dict(scale=1.0 + rng.rand(dim),
                               offset=rng.rand(dim)))

        f, finv, f_jac_logD, finv_jac_logD = named_bijection(name, **kwargs)

        x = rng.rand(nsamples, dim)  # values between 0 and 1
        y = f(x)
        assert np.allclose(x, finv(y), atol=1e-8)
        assert np.allclose(x[0], finv(y[0]), atol=1e-8)
        assert np.allclose(f_jac_logD(x), -finv_jac_logD(y), atol=1e-8)
        assert np.allclose(f_jac_logD(x[0]), -finv_jac_logD(y[0]), atol=1e-8)


def test_batched_matrix_ops(dim=4, nsamples=100):
    A_pd = np.full((nsamples, dim, dim), np.nan, dtype=dtype)
    A_nonsing = np.full((nsamples, dim, dim), np.nan, dtype=dtype)
    L = np.full((nsamples, dim, dim), np.nan, dtype=dtype)
    inv = np.full((nsamples, dim, dim), np.nan, dtype=dtype)
    det = np.full(nsamples, np.nan, dtype=dtype)
    for i in range(nsamples):
        L[i] = np.tril(np.random.randn(dim, dim), -1) + np.diag(1.0 + np.exp(np.random.randn(dim)))
        A_pd[i] = np.dot(L[i], L[i].T)
        L2 = np.tril(np.random.rand(dim, dim), -1) + np.diag(np.exp(np.random.randn(dim)))
        L3 = np.tril(np.random.rand(dim, dim), -1) + np.diag(np.exp(np.random.randn(dim)))
        A_nonsing[i] = np.dot(np.dot(L2, A_pd[i]), L3.T)
        inv[i] = np.linalg.inv(A_nonsing[i])
        det[i] = np.linalg.det(A_nonsing[i])

    tA = sym.tensorN(3)
    f_choleach = theano.function(inputs=[tA], outputs=sym.cholesky_each(tA))
    f_inveach = theano.function(inputs=[tA], outputs=sym.invert_each(tA))
    f_deteach = theano.function(inputs=[tA], outputs=sym.det_each(tA))

    symL = f_choleach(A_pd)
    symdet = f_deteach(A_nonsing)
    syminv = f_inveach(A_nonsing)

    assert np.allclose(symL, L, atol=1e-8)
    assert np.allclose(symdet, det, atol=1e-8)
    assert np.allclose(syminv, inv, atol=1e-8)

    try:
        f_choleach(A_nonsing)  # try Cholesky factorizing some non-symmetric matrices
    except Exception as e:
        assert isinstance(e, np.linalg.linalg.LinAlgError), \
            "unexpected error when trying Cholesky factorization of non-symmetric matrix"
