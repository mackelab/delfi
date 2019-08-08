import numpy as np
import delfi.utils.math as mm
from delfi.distribution import Gaussian
from delfi.utils.bijection import named_bijection


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
        assert np.allclose(f_jac_logD(x), -finv_jac_logD(y), atol=1e-8)