import theano
import theano.tensor as tt
import theano.tensor.slinalg as slinalg
import numpy as np


def MyLogSumExp(x, axis=None):
    x_max = tt.max(x, axis=axis, keepdims=True)
    return tt.log(tt.sum(tt.exp(x - x_max), axis=axis, keepdims=True)) + x_max


def mog_LL(y, la, ms, Ps, ldetPs, from_cholesky=False):
    """
    Calculate log-likelihood of data points y for a MoG

    Parameters
    ----------
    y : 2D tensor (N_data x N_dim)
        data points at which to evaluate the MoG
    la : 2D tensor (N_data x N_components)
        logs of mixture coefficients
    ms : N_components-element list of 2D tensors (N_data x N_dim)
        Mixture means
    Ps : N_components-element list of 3D tensors (N_data x N_dim x N_dim)
        Mixture precisions
    ldetPs : N_components-element list of 1D tensors (N_data)
        Mixture precision log determinants
    from_cholesky: bool
        Whether to instead interpret Ps as Cholesky factors of precisions,
        such that the precision of the j-th component for the -th data point is
        dot(Ps[j][i,:,:].T, Ps[j][i,:,:])

    Returns
    -------
    LL : 1D tensor (N_data)
        log-likelihood of MoG at the provided data points
    """
    # data likelihood for each Gaussian component (except for scale factor)
    if from_cholesky:
        lps = [-0.5 * tensorQF_chol(P, y - m) + ldetP
               for m, P, ldetP in zip(ms, Ps, ldetPs)]
    else:
        lps = [-0.5 * (tensorQF(P, y - m) - ldetP)
               for m, P, ldetP in zip(ms, Ps, ldetPs)]

    # include missing scale factor
    logpdf_offset = -0.5 * np.log(2 * np.pi) * y.shape[1]

    # add up components' Gauss pdfs, apply missing factor, stay in log domain
    return MyLogSumExp(tt.stack(lps, axis=1) + la, axis=1) + logpdf_offset


def invert_each(A):
    """
    Invert a set of matrices stored in an N-D array. The rows and columns of
    each matrix must correspond to the last 2 dimensions of the array, which
    must be equal.
    """
    return batched_matrix_op(A, tt.nlinalg.MatrixInverse(), 2)


def det_each(A, log=False):
    """
    Calculate determinants for a set of matrices stored in an N-D array. The
    rows and columns of each matrix must correspond to the last 2 dimensions of
    the array, which must be equal.
    """
    D = batched_matrix_op(A, tt.nlinalg.Det(), 0)
    return tt.log(D) if log else D


def cholesky_each(A):
    """
    Calculate cholesky factorizations for a set of positive definite matrices.

    :param A: Array of input matrices. The rows and columns of each matrix must correspond to the last 2 dimensions of
    A, which must be equal.
    :return: cholesky factors
    """
    return batched_matrix_op(A, slinalg.Cholesky(), 2)


def batched_matrix_op(A, Op, Op_output_ndim, allow_gc=False):
    """
    Apply a unary operator to a set of matrices stored in an N-D array (N > 2).
    The rows and columns of each matrix must correspond to the last 2
    dimensions of the array. Op_output_ndim should be 0 for a scalar output, 1
    for a vector, 2 for a matrix, etc.
    """
    sB = (tt.prod(A.shape[:-2]), A.shape[-2], A.shape[-1])
    B = A.reshape(sB, ndim=3)
    OpB, _ = theano.scan(fn=lambda X: Op(X), allow_gc=allow_gc, sequences=B)
    ndim_out = (A.ndim - 2) + Op_output_ndim
    sOpA = tt.join(0, A.shape[:-2], OpB.shape[1:])
    return OpB.reshape(sOpA, ndim=ndim_out)


def tensorN(N, name=None, dtype=theano.config.floatX):
    """
    Return a tensor of the specified dimension.
    """
    if N == 1:
        return tt.vector(name=name, dtype=dtype)
    if N == 2:
        return tt.matrix(name=name, dtype=dtype)
    elif N == 3:
        return tt.tensor3(name=name, dtype=dtype)
    elif N == 4:
        return tt.tensor4(name=name, dtype=dtype)
    else:
        raise NotImplementedError


def tensorQF(A, x):
    """
    Symbolically evaluate quadratic form with matrix A on vector x.

    The first dimensions of A and x range over "data points", in the sense that
    dot(dot(x[i, :], A[i, :, :]), x[i, :]) is evaluated for each i.

    The returned tensor will be 1 dimensional
    """
    return tt.sum(tt.sum(x.dimshuffle([0, 'x', 1]) * A, axis=2) * x, axis=1)


def tensorQF_chol(U, x):
    """
    Symbolicaly evaluate quadratic form with matrix dot(U^T, U) on vector x.
    This will usually be called on a U resulting from a Cholesky factorization.
    See tensorQF for further details.
    """
    return tt.sum(tt.sum(x.dimshuffle([0, 'x', 1]) * U, axis=2) ** 2, axis=1)
