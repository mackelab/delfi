import itertools
import numpy as np
from delfi.distribution import MoG


def MoGL2sq(p1, p2):
    """
    Calculate integral of squared difference between MoG pdfs

    idea from this stats.stackexchange post:
    https://stats.stackexchange.com/questions/71879/distance-between-two-gaussian-mixtures-to-evaluate-cluster-solutions
    """
    assert isinstance(p1, MoG) and isinstance(p2, MoG)

    w = np.concatenate((p1.a, -p2.a))
    allxs = p1.xs + p2.xs

    L2 = 0.0
    # compute all the integrals. don't compute cross terms twice
    for i in range(len(w)):
        L2 += w[i] ** 2 * gaussprodintegral(allxs[i], allxs[i])
        for j in range(i):
            L2 += 2 * w[i] * w[j] * gaussprodintegral(allxs[i], allxs[j])

    return L2


def gaussprodintegral(x1, x2, log=False):
    """
    Calculate normalization constant for a product of Gaussian pdfs.
    This is the integral of the unnormalized product.
    """
    d = x1.m - x2.m
    S = x1.S + x2.S

    P = np.linalg.inv(S)
    logdetS = np.log(np.linalg.det(S))
    Q = np.sum(np.dot(d, P) * d)  # quadratic term in Gaussian pdf exponent

    logZ = 0.5 * (-Q - logdetS - x1.ndim * np.log(2.0 * np.pi))

    res = logZ if log else np.exp(logZ)
    return res


def gaussquotientintegral(x1, x2, log=False):
    """
    Calculate normalization constant for a quotient of Gaussian pdfs.
    This is the integral of the unnormalized product.
    """
    Pstar = x1.P - x2.P
    Sstar = np.linalg.inv(Pstar)
    detSstar = np.linalg.det(Sstar)
    if detSstar < 0:
        return np.inf
    logdetSstar = np.log(detSstar)
    Pstarmstar = x1.Pm - x2.Pm
    mstar = np.dot(Sstar, Pstarmstar)

    Q = np.dot(x1.Pm, x1.m) - np.dot(x2.Pm, x2.m) - np.dot(Pstarmstar, mstar)
    D = x1.logdetP - x2.logdetP + logdetSstar + x1.ndim * np.log(2.0 * np.pi)
    logZ = 0.5 * (D - Q)

    res = logZ if log else np.exp(logZ)
    return res


def gaussprodquotientintegral(x1, x2, x3, log=False):
    """
    Calculate normalization constant for a product-quotient of 3 Gaussian pdfs.
    If x1, x2 and x3 are Gaussian with pdfs f1, f2 and f3, then we integrate
    f1 * f2 / f3 and return the resulting normalization constant.
    """
    Pstar = x1.P + x2.P - x3.P
    Sstar = np.linalg.inv(Pstar)
    detSstar = np.linalg.det(Sstar)
    if detSstar < 0:
        return np.inf
    logdetSstar = np.log(detSstar)
    Pstarmstar = x1.Pm + x2.Pm - x3.Pm
    mstar = np.dot(Sstar, Pstarmstar)

    Q = np.dot(x1.Pm, x1.m) + np.dot(x2.Pm, x2.m) \
        - np.dot(x3.Pm, x3.m) - np.dot(Pstarmstar, mstar)
    D = x1.logdetP + x2.logdetP - x3.logdetP + logdetSstar
    logZ = 0.5 * (D - Q)

    res = logZ if log else np.exp(logZ)
    return res
