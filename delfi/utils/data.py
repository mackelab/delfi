import numpy as np


def isint(x):
    return np.issubdtype(type(x), np.integer)


def repnewax(A, n, axis=0):
    """Add a new axis (default 0), shifting other dimensions down. Then repeat
    all values on this new axis n times"""
    return np.expand_dims(A, axis).repeat(n, axis=axis)


def combine_trn_datasets(trn_datasets, max_inputs=None):
    n_vars = len(trn_datasets[0])
    if max_inputs is not None:
        n_vars = np.minimum(n_vars, max_inputs)
    td = [np.concatenate([x[j] for x in trn_datasets], axis=0)
          for j in range(n_vars)]
    return tuple(td)