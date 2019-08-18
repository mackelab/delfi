import numpy as np
from functools import partial
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


def affine_map(x, s, o):
    return x * s + o


def inv_affine_map(y, s, o):
    return (y - o) / s


def affine_jac_logD(x, s):
    if x.ndim == 1:
        return np.log(s).sum()
    return np.full(x.shape[:-1], np.log(s).sum())


def inv_affine_jac_logD(y, s):
    if y.ndim == 1:
        return -np.log(s).sum()
    return np.full(y.shape[:-1], -np.log(s).sum())


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

        f = partial(affine_map, s=s, o=o)
        finv = partial(inv_affine_map, s=s, o=o)
        f_jac_logD = partial(affine_jac_logD, s=s)
        finv_jac_logD = partial(inv_affine_jac_logD, s=s)

    elif name == 'norminvcdf':

        f = ndtri
        finv = ndtr
        f_jac_logD = norminvcdf_jac_logD
        finv_jac_logD = normcdf_jac_logD

    else:
        raise ValueError('unknown bijection: {0}'.format(name))

    return f, finv, f_jac_logD, finv_jac_logD
