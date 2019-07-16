import numpy as np
from delfi.distribution.BaseDistribution import BaseDistribution
#from delfi.distribution.mixture.BaseMixture import BaseMixture
#from delfi.neuralnet.NeuralNet import MAFconditional
from copy import deepcopy


class TransformedDistribution(BaseDistribution):
    """Distribution object that carries out an invertible change of variables
    for another distribution object.

    A bijection must be supplied mapping from
    the original distributions variables into the new one's along with the
    bijection's inverse and the log determinant of the bijection's Jacobian.

    There is no checking of whether the supplied functions are really inverses
    of each other or are in fact bijections at all, this is up to the user.

    Parameters
    ----------
    distribution : delfi distribution or mixture object
        Original distrib. to be transformed. Must implement eval() and gen()
    bijection : callable
        Bijective mapping from original distrib.'s random variable to this one's
    inverse_bijection: callable
        Inverse of the bijective mapping, going from this distribution's random
        variable to the original one's.
    bijection_jac_logD: callable
        Log determinant of the Jacobian of the bijection from the original distribution's random variable to this one's.
    makecopy: bool
        Whether to call deepcopy on the simulator, unlinking the RNGs
    """
    def __init__(self, distribution, bijection, inverse_bijection, bijection_jac_logD, makecopy=False):
        #assert isinstance(distribution, BaseDistribution) or isinstance(distribution, BaseMixture) \
        #    or isinstance(distribution, MAFconditional)
        if makecopy:
            distribution = deepcopy(distribution)
        self.distribution = distribution
        self.bijection, self.inverse_bijection = bijection, inverse_bijection
        self.bijection_jac_logD = bijection_jac_logD
        self.ndim = distribution.ndim

    @property
    def mean(self):
        """Means"""
        return np.nan(self.ndim)  # generally unknown

    @property
    def std(self):
        """Standard deviations of marginals"""
        return np.nan(self.ndim)  # generally unknown

    @copy_ancestor_docstring
    def eval(self, x, ii=None, log=True):
        assert ii is None, "cannot marginalize transformed distributions"
        x_original = self.inverse_bijection(x)
        logp_original = self.distribution.eval(x_original, log=True)
        logp = logp_original - self.bijection_jac_logD(x_original)  # change of variables
        return logp if log else np.exp(logp)

    @copy_ancestor_docstring
    def gen(self, n_samples=1):
        samples = self.distribution.gen(n_samples=n_samples)
        return self.bijection(samples)

    def reseed(self, seed):
        """Reseeds the distribution's RNG"""
        self.distribution.reseed(seed)

    def gen_newseed(self):
        """Generates a new random seed"""
        return self.distribution.gen_newseed()
