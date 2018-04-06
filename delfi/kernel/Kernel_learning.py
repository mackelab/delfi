import lasagne
import numpy as np
import theano
import theano.tensor as T
import lasagne.updates as lu

import abc
from delfi.utils.meta import ABCMetaDoc
from delfi.neuralnet.Trainer import Trainer

class KernelLayer(lasagne.layers.Layer):
    def __init__(self, incoming, B=lasagne.init.Normal(0.01), **kwargs):
        super(KernelLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.eye = T.eye(num_inputs)
        self.B = self.add_param(B, (num_inputs, ), name='B')

    def get_output_for(self, input, **kwargs):
        D = T.dot(self.B*self.eye, self.B*self.eye.T)
        inner = (input.dot(D)*input).sum(axis=1)
        return T.exp(-inner)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],)

    
class KernelLayer_offset(lasagne.layers.Layer):
    def __init__(self, incoming, B=lasagne.init.Normal(0.01), Z=lasagne.init.Normal(0.01), **kwargs):
        super(KernelLayer_offset, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.eye = T.eye(num_inputs)
        self.B = self.add_param(B, (num_inputs, ), name='B')
        self.Z = self.add_param(Z, (1,), name='Z')

    def get_output_for(self, input, **kwargs):
        D = T.dot(self.B*self.eye, self.B*self.eye.T)
        inner = (input.dot(D)*input).sum(axis=1)
        return T.exp(-inner + self.Z)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],)    
        
    
class KernelLayer_full(lasagne.layers.Layer):
    def __init__(self, incoming, B=lasagne.init.Normal(0.01), Z=lasagne.init.Normal(0.01), **kwargs):
        super(KernelLayer_full, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        #self.eye = T.eye(num_inputs)
        self.B = self.add_param(B, (num_inputs, num_inputs), name='B')
        self.Z = self.add_param(Z, (1,), name='Z')

    def get_output_for(self, input, **kwargs):
        D = T.dot(self.B, self.B.T)
        inner = (input.dot(D)*input).sum(axis=1)
        return T.exp(-inner + self.Z)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],)    


class My_Helper_Kernel(metaclass=ABCMetaDoc):
    def __init__(self, obs, A, Z):
        """Temporary substitute for working Gaussian kernel class in delfi
        Parameters
        ----------
        obs : 1 x dim
            center of kernel
        A : dim x dim or dim
            kernel matrix (cf. precision matrix for Gaussian distributions)
        Z : kernel log-normalizer
        -----
        Notes:
        k(x) = exp( - (x-obs)' A (x-obs) + Z )
        """

        assert obs.shape[0] == 1, 'obs.shape[0] must be 1'
        assert obs.shape[1] >= 1, 'obs.shape[1] must be >= 1'

        self.dim = obs.shape[1]
        self.obs = obs

        self.diag_A = False if A.ndim>1 else True        
        self.A = A
        self.Z = Z

        if self.diag_A:
            self.B = np.sqrt(A)
        else:
            self.B = np.linalg.cholesky(A)

    @abc.abstractmethod
    def kernel(u):
        pass        


    def eval(self, x):
        """Kernel for loss calibration

        Parameters
        ----------
        x : N x dim
            points at which to evaluate kernel

        Returns
        -------
        weights : N
            normalized to be 1. for x = obs
        """
        assert x.shape[0] >= 1, 'x.shape[0] needs to be >= 1'
        assert x.shape[1] == self.dim, 'x.shape[1] needs to be == self.obs'

        if self.diag_A:
            z = (x-self.obs) * self.B.reshape(1,-1)
            out = np.exp( -(z*z).sum(axis=1) + self.Z)
        else:
            z = (x-self.obs).dot(self.B)
            out = np.exp( -(z*z).sum(axis=1) + self.Z)

        return out

def kernel_opt(iws, stats, obs, kernel_loss=None, step=lu.adam,
               epochs=100, minibatch=100, lr=0.001, lr_decay=1.0, 
               max_norm=0.1, monitor=None, monitor_every=None, 
               stop_on_nan=False, verbose=False, seed=None):

    assert kernel_loss in (None, 'x_kl', 'ess')

    dtype = theano.config.floatX
    input_var = T.matrix('inputs', dtype=dtype)
    target_var = T.vector('targets', dtype=dtype)

    dx = (stats - obs).astype(np.float32)

    # set up learning model 
    l_in = lasagne.layers.InputLayer(shape=(None,obs.size),input_var=input_var)
    l_dot = KernelLayer_offset(l_in, name='kernel_layer')
    prediction = lasagne.layers.get_output(l_dot)
    w_opt = prediction * target_var
    
    if kernel_loss == 'ess':
        loss =  - T.sum(w_opt)**2 / T.sum(w_opt**2)
    elif kernel_loss == 'x_kl':
        loss = T.log( T.mean(w_opt) ) - T.mean(T.log(prediction))        
    elif kernel_loss is None:
        return My_Helper_Kernel(obs=obs, A=np.zeros(obs.size), Z=0.)
    else:
        raise NotImplementedError
    
    params_k = lasagne.layers.get_all_params(l_dot, trainable=True)

    class Kernel(object):
        def __init__(self, params):
            self.aps = params            

    krnl = Kernel(params_k)                   # this is dumb

    trnr = Trainer(krnl, loss, 
                   trn_data=(dx, iws), 
                   trn_inputs=[input_var, target_var],
                   step=step, lr=lr, lr_decay=lr_decay, max_norm=None,
                   monitor=monitor, seed=seed)

    logs = trnr.train(epochs=epochs,
               minibatch=minibatch,
               monitor_every=monitor_every,
               stop_on_nan=stop_on_nan,
               verbose=verbose)    

    B = krnl.aps[0].get_value()
    A = B*B if B.ndim==1 else B.dot(B.T)

    Z = krnl.aps[1].get_value()

    cbkrnl = My_Helper_Kernel(obs=obs, A=A, Z=Z)

    return cbkrnl, logs['loss']