import abc
import numpy as np

from delfi.utils.meta import ABCMetaDoc
from delfi.utils.progress import no_tqdm, progressbar


class BaseSimulator(metaclass=ABCMetaDoc):
    def __init__(self, dim_param, seed=None):
        """Abstract base class for simulator models

        Simulator models must at least implement abstract methods and properties
        of this class.

        Parameters
        ----------
        dim_param : int
            Dimensionality of parameter vector
        seed : int or None
            If provided, random number generator will be seeded
        """
        self.dim_param = dim_param

        self.rng = np.random.RandomState(seed=seed)
        self.seed = seed

    def gen(self, params_list, n_reps=1, pbar=None):
        """Forward model for simulator for list of parameters

        Parameters
        ----------
        params_list : list of lists or 1-d np.arrays
            List of parameter vectors, each of which will be simulated
        n_reps : int
            If greater than 1, generate multiple samples given param
        pbar : tqdm.tqdm or None
            If None, will do nothing. Otherwise it will call pbar.update(1)
            after each sample.

        Returns
        -------
        data_list : list of lists containing n_reps dicts with data
            Repetitions are runs with the same parameter set, different
            repetitions. Each dictionary must contain a key data that contains
            the results of the forward run. Additional entries can be present.
        """
        data_list = []
        for param in params_list:
            rep_list = []
            for r in range(n_reps):
                rep_list.append(self.gen_single(param))
            data_list.append(rep_list)
            if pbar is not None:
                pbar.update(1)

        return data_list

    @abc.abstractmethod
    def gen_single(self, params):
        """Forward model for simulator for single parameter set

        Parameters
        ----------
        params : list or np.array, 1d of length dim_param
            Parameter vector

        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of
            the forward run. Additional entries can be present.
        """
        pass

    def gen_newseed(self):
        """Generates a new random seed"""
        if self.seed is None:
            return None
        else:
            return self.rng.randint(0, 2**31)
