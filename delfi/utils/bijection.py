import numpy as np
from scipy.special import expit, logit, ndtr, ndtri


def expit_jac_logD(y):
    x = expit(y)
    return (np.log(x) + np.log(1 - x)).sum(axis=-1)


def logit_jac_logD(x):
    return -(np.log(x) + np.log(1 - x)).sum(axis=-1)


def normcdf_jac_logD(y):  # log of Gaussian pdf
    return -0.5 * (y ** 2 + np.log(2.0 * np.pi)).sum(axis=-1)


def norminvcdf_jac_logD(x):
    return -normcdf_jac_logD(ndtri(x))


def named_bijection(name, **kwargs):
    name = name.lower()

    if name == 'logit':

        f = logit
        finv = expit
        f_jac_logD = logit_jac_logD
        finv_jac_logD = expit_jac_logD

    elif name == 'affine':

        s = kwargs['scale'].copy()
        o = kwargs['offset'].copy()

        def f(x):
            return x * s + o

        def finv(y):
            return (y - o) / s

        def f_jac_logD(x):
            return np.log(s).sum(axis=-1)

        def finv_jac_logD(y):
            return -np.log(s).sum(axis=-1)

    elif name == 'norminvcdf':

        f = ndtri
        finv = ndtr
        f_jac_logD = norminvcdf_jac_logD
        finv_jac_logD = normcdf_jac_logD

    else:
        raise ValueError('unknown bijection: {0}'.format(name))

    return f, finv, f_jac_logD, finv_jac_logD
