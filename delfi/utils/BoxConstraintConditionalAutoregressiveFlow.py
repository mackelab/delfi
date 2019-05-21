from snl.ml.models.mafs import ConditionalMaskedAutoregressiveFlow
import numpy as np
import theano.tensor as tt
import theano

from delfi.neuralnet.NeuralNet import dtype
#from delfi.utils.symbolic import tensorN, MyLogSumExp

def sym_sig_inv(x):

	return tt.log( x / ( 1. - x ) ) # there's probably more stable implementations...

def sig(x):

	return 1. / (1. + np.exp( - x )) # there's probably more stable implementations...


class BoxConstraintConditionalAutoregressiveFlow(ConditionalMaskedAutoregressiveFlow):

	def __init__(self, n_inputs, n_outputs, n_hiddens, act_fun, n_mades, 
		upper, lower, 
		batch_norm=True, output_order='sequential', mode='sequential', 
		input=None, output=None, rng=np.random):

		y = tt.matrix('y', dtype=dtype) if output is None else output		

		self.upper, self.lower = upper, lower # add some checks
		self.diff = self.upper - self.lower

		y_scaled = (y - self.lower) / self.diff
		y_sig = sym_sig_inv(y_scaled) # (logit-) transformed y

		super().__init__(n_inputs=n_inputs, n_outputs=n_outputs, 
			n_hiddens=n_hiddens, act_fun=act_fun, n_mades=n_mades, 
			batch_norm=batch_norm, output_order=output_order, mode=mode, 
			input=input, output=y_sig, rng=rng)

		L = theano.clone(output=self.L, replace={self.y:y_sig}, share_inputs=True)	
		self.L = L - tt.sum(tt.log(self.diff * y_scaled * (1. - y_scaled)), axis=1)
		self.L.name = 'L'

		self.y = y


	def gen(self, x, n_samples=None, u=None, rng=np.random): # overwrite that one

		y = super().gen(x=x, n_samples=n_samples, u=u, rng=rng)

		return self.diff * sig(y) + self.lower
