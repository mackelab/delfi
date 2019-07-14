from copy import deepcopy

from delfi.simulator.BaseSimulator import BaseSimulator


class TransformedSimulator(BaseSimulator):
    def __init__(self, simulator, inverse_bijection, makecopy=False):
        '''
        Simulator with parameters in a transformed space. An inverse bijection
        must be supplied to map back into the original parameter space. This
        reparamterization allows unrestricted real-valued Euclidean parameter
        spaces for simulators whose outputs are defined only for certain
        parameter values

        For example, a log transform can make positive numbers onto the real
        line, and a logit transform can map the unit interval onto the real
        line. In each case, the inverse bijection (e.g. exp or logisitic) must
        be supplied.

        There is no checking that the user-supplied bijection inverse is in fact
        a one-to-one mapping, this is up to the user to verify.

        :param simulator: Original simulator
        :param inverse_bijection: Inverse transformation back into original simulator's parameter space
        :param makecopy: Whether to call deepcopy on the simulator, unlinking the RNGs
        '''
        if makecopy:
            simulator = deepcopy(simulator)
        self.simulator, self.inverse_bijection = simulator, inverse_bijection
        self.dim_param = self.simulator.dim_param

    def reseed(self, seed):
        self.simulator.reseed(seed)

    def gen_newseed(self):
        return self.simulator.gen_newseed()

    def gen_single(self, input_params):
        transformed_params = self.inverse_bijection(input_params)
        return self.simulator.gen_single(transformed_params)