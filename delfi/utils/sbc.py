import numpy as np

class SBC(object):
    def __init__(self, generator, inf, f):
        """ Simulation-based calibration

        Parameters
        ----------
        generator : generator instance
            Generator instance
        inf : inference instance
            Infernece instance
        f : test-function
            Test-function (maps parameters into test statistics).
            Has to be appliable to N x dim(theta) data matrices.
        """
        self.generator = generator # delfi generator object
        self.inf = inf             # delfi inference object
        self.f = f                 # test-function (maps x->f(x))
        
    def sample_full(self, N):
        out = self.generator.gen(N) # will sample from generator.proposal unless it's None
        return out[0], out[1]

    def get_conditional(self, x):
        return self.inf.predict_uncorrected(x)
    
    def test(self, N, L, data=None):

        data = self.sample_full(N)  if data is None else data
        assert data[0].ndim == 2        
        N = data[0].shape[0]

        dim = self.f(data[0][:1,:]).size
        
        res  = np.empty((N, dim))
        
        for i in range(N):
            f0 = self.f(data[0][i,:]).reshape(1,-1)
            p = self.get_conditional(data[1][i,:])
            
            batch = self.f(p.gen(L))
            assert batch.shape==(L, f0.size)
                
            res[i,:] = np.sum( f0 < batch , axis=0)

        return res
