import delfi.distribution as dd
import numpy as np
from delfi.simulator.BaseSimulator import BaseSimulator


def default_mapfunc(theta, p):
    ang = -np.pi / 4.0
    c = np.cos(ang)
    s = np.sin(ang)
    z0 = c * theta[0] - s * theta[1]
    z1 = s * theta[0] + c * theta[1]
    return p + np.array([-np.abs(z0), z1])


def default_mapfunc_inverse(theta, x):
    ang = -np.pi / 4.0
    c = np.cos(ang)
    s = np.sin(ang)
    z0 = c * theta[0] - s * theta[1]
    z1 = s * theta[0] + c * theta[1]
    return x - np.array([-np.abs(z0), z1])


def default_mapfunc_Jacobian_determinant(theta, p):
    # det. of Jacobian from p to x
    return 1.0


class TwoMoons(BaseSimulator):
    def __init__(self, mean_radius=1.0, sd_radius=0.1, baseoffset=1.0,
                 mapfunc=None, mapfunc_inverse=None, mapfunc_Jacobian_determinant=None,  # transforms noise dist.
                 seed=None):
        """Two Moons simulator

        Toy model that draws data from a crescent shaped mixture distribution.
        For the default mapfunc, this leads to a bimodal posterior, with each
        mode the same shape as the simulator's data density.

        Parameters
        ----------
        mean_radius: float
            Radius of curvature for each moon in the posterior
        sd_radius: float
            Dispersion of samples perpendicular to moon curvature
        base_offset: float
            Minimum separation between moons in the posterior
        mapfunc: callable or None
            Mapping applied to points. Default as described in Greenberg et al., 2019
        mapfunc_inverse: callable or None
            Inverse of mapping
        mapfunc_Jacobian_determinant: callable or None
            determinant of Jacobian of manfunc, used for change of variables when calculating likelihood
        seed : int or None
            If set, randomness is seeded
        """
        super().__init__(dim_param=2, seed=seed)
        self.mean_radius = mean_radius
        self.sd_radius = sd_radius
        self.baseoffset = baseoffset
        if mapfunc is None:
            self.mapfunc = default_mapfunc
            self.mapfunc_inverse = default_mapfunc_inverse
            self.mapfunc_Jacobian_determinant = default_mapfunc_Jacobian_determinant
        else:
            self.mapfunc, self.mapfunc_inverse, self.mapfunc_Jacobian_determinant = \
                mapfunc, mapfunc_inverse, mapfunc_Jacobian_determinant

    @copy_ancestor_docstring
    def gen_single(self, param):
        # See BaseSimulator for docstring
        param = np.asarray(param).reshape(-1)
        assert param.ndim == 1
        assert param.shape[0] == self.dim_param

        a = np.pi * (self.rng.rand() - 0.5)
        r = self.mean_radius + self.rng.randn() * self.sd_radius
        p = np.array([r * np.cos(a) + self.baseoffset, r * np.sin(a)])
        return {'data': self.mapfunc(param, p)}

    def likelihood(self, param, x, log=True):
        assert x.size == 2, "not yet implemented for evaluation on multiple points at once"
        assert np.isfinite(x).all() and (np.imag((x)) == 0).all(), "invalid input"
        if self.mapfunc_inverse is None or self.mapfunc_Jacobian_determinant is None:
            return np.nan
        p = self.mapfunc_inverse(param, x)
        assert p.size == 2, "not yet implemented for non-bijective map functions"
        u = p[0] - self.baseoffset
        v = p[1]

        if u < 0.0:  # invalid x for this theta
            return -np.inf if log else 0.0

        r = np.sqrt(u ** 2 + v ** 2)  # note the angle distribution is uniform
        L = -0.5 * ((r - self.mean_radius) / self.sd_radius) ** 2 - 0.5 * np.log(2 * np.pi * self.sd_radius ** 2)
        return L if log else np.exp(L)

    def gen_posterior_samples(self, obs=np.array([0.0, 0.0]), prior=None, n_samples=1):
        # works only when we use the default_mapfunc above

        # use opposite rotation as above
        ang = -np.pi / 4.0
        c = np.cos(-ang)
        s = np.sin(-ang)

        theta = np.zeros((n_samples, 2))
        for i in range(n_samples):
            p = self.gen_single(np.zeros(2))['data']
            q = np.zeros(2)
            q[0] = p[0] - obs[0]
            q[1] = obs[1] - p[1]

            if np.random.rand() < 0.5:
                q[0] = -q[0]

            theta[i, 0] = c * q[0] - s * q[1]
            theta[i, 1] = s * q[0] + c * q[1]

        return theta
