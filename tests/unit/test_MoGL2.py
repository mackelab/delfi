import numpy as np
import scipy.stats
from delfi.distribution import MoG
from delfi.utils.math import MoGL2sq


def test_1D_1K():
    m1 = 0.0
    m2 = 1.0
    S1 = 1.0
    S2 = 2.0

    # first MoG
    q1 = MoG(a=[1.0], ms=[[m1]], Ss=[[[S1]]])

    # second MoG
    q2 = MoG(a=[1.0], ms=[[m2]], Ss=[[[S2]]])

    A = scipy.stats.norm.pdf(m1, m1, np.sqrt(S1 + S1))
    B = scipy.stats.norm.pdf(m2, m2, np.sqrt(S2 + S2))
    C = scipy.stats.norm.pdf(m1, m2, np.sqrt(S1 + S2))
    assert np.isclose(MoGL2sq(q1, q2), A + B - 2 * C)


def test_2D_2K():
    # first MoG
    mu1 = np.array([0., 0.])
    mu2 = np.array([0.25, 0.5])
    S1 = np.array([[1., 0.], [0., 1.]])
    S2 = np.array([[.5, .1], [.1, .5]])
    q12 = MoG(a=[.5, .5], ms=[mu1, mu2], Ss = [S1, S2])

    # second MoG
    mu3 = np.array([-0.1, 0.25])
    mu4 = np.array([0.3, 0.4])
    S3 = np.array([[1.1, 0.], [0., 0.9]])
    S4 = np.array([[.45, .1], [.1, .55]])
    q34 = MoG(a=[.45, .55], ms=[mu3, mu4], Ss = [S3, S4])

    assert MoGL2sq(q12, q34) > 0
    assert np.isclose(MoGL2sq(q12, q12), 0)
