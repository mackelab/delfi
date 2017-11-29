import collections
import dill as pickle
import fnmatch
import os

from delfi.neuralnet.NeuralNet import NeuralNet


def load(file):
    """Loads inference instance from pickle

    Parameters
    ----------
    file : str

    Returns
    -------
    inference instance
    """
    data = load_pkl(file)
    if 'delfi.inference' in str(type(data['inference'])):
        inference = data['inference']
        inference.compile_observables()
        inference.network.compile_funs()
        return inference
    else:
        raise NotImplementedError


def load_pkl(file):
    """Loads data from pickle

    Parameters
    ----------
    file : str

    Returns
    -------
    data
    """
    f = open(file, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def save(obj, file):
    """Saves inference result

    Parameters
    ----------
    obj : inference instance
    file : str
    """
    data = {}
    if 'delfi.inference' in str(type(obj)):
        data['generator'] = obj.generator
        data['network.spec_dict'] = obj.network.spec_dict
        data['network.params_dict'] = obj.network.params_dict
        del obj.observables
        data['inference'] = obj
        save_pkl(data, file)
        obj.compile_observables()
    else:
        raise NotImplementedError


def save_pkl(data, file):
    """Saves data to a pickle

    Parameters
    ----------
    data : object
    file : str
    """
    f = open(file, 'wb')
    pickle.dump(data, f)
    f.close()
