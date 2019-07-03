import numpy as np

from delfi.simulator.BaseSimulator import BaseSimulator


class TransformedSimulator(BaseSimulator):
    def __init__(self, simulator, inverse_bijection):
        '''
        Simulator with parameters in a transformed

        :param simulator: Original simulator
        :param inverse_bijection: Inverse transformation back into original simulator's parameter space
        '''
        self.simulator, self.inverse_bijection = simulator, inverse_bijection
        self.dim_param = self.simulator.dim_param

    def reseed(self, seed):
        self.simulator.reseed(seed)

    def gen_newseed(self):
        return self.simulator.gen_newseed()

    def gen_single(self, input_params):
        transformed_params = self.inverse_bijection(input_params)
        return self.simulator.gen_single(transformed_params)