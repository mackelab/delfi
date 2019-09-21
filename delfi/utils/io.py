import dill as pickle
import os
from importlib.util import spec_from_file_location, module_from_spec
from importlib import import_module


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


def import_from_module_and_run(module_name, function_name, *args, starting_path=None, **kwargs):
    """

    :param module_name:
    :param function_name:
    :param args:
    :param starting_path:
    :param kwargs:
    :return:
    """
    assert isinstance(function_name, str) and isinstance(module_name, str)
    if starting_path is not None:
        prev_dir = os.getcwd()
        os.chdir(starting_path)

    try:
        m = import_module(module_name)
        f = getattr(m, function_name)
        result = f(*args, **kwargs)
        err = None
    except Exception as e:
        err = e
    finally:
        if starting_path is not None:
            os.chdir(prev_dir)

    if err is not None:
        raise err

    return result


def run_function_from_file(file, function_name, *args, starting_path=None, **kwargs):
    """
    Utility to retrieve a function by name from a python file and run it with optional inputs.

    This can be useful for running simulators on remote hosts with installing/importing the necessary modules on the
    client.
    :param starting_path:
    :param function_name:
    :param file:
    :param function_name:
    :param args:
    :param kwargs:
    :return:
    """
    assert isinstance(function_name, str)
    assert os.path.exists(file), "file not found: {0}".format(file)
    if starting_path is not None:
        prev_dir = os.getcwd()
        os.chdir(starting_path)

    try:
        spec = spec_from_file_location('rfff_module', file)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        f = eval('module.{0}'.format(function_name))
        success = True
    except Exception as e:
        success = False
        err = e

    if starting_path is not None:
        os.chdir(prev_dir)

    if not success:
        raise err

    return f(*args, **kwargs)
