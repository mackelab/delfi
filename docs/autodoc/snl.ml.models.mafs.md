## **ConditionalMaskedAutoregressiveFlow**`#!py3 class` { #ConditionalMaskedAutoregressiveFlow data-toc-label=ConditionalMaskedAutoregressiveFlow }


```
Implements a Conditional Masked Autoregressive Flow.
```

### *ConditionalMaskedAutoregressiveFlow*.**\_\_init\_\_**`#!py3 (self, n_inputs, n_outputs, n_hiddens, act_fun, n_mades, batch_norm=True, output_order='sequential', mode='sequential', input=None, output=None, rng=<module 'numpy.random' from '/Users/jm/opt/anaconda3/envs/ind/lib/python3.7/site-packages/numpy/random/__init__.py'>)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Constructor.
:param n_inputs: number of (conditional) inputs
:param n_outputs: number of outputs
:param n_hiddens: list with number of hidden units for each hidden layer
:param act_fun: name of activation function
:param n_mades: number of mades in the flow
:param batch_norm: whether to use batch normalization between mades in the flow
:param output_order: order of outputs of last made
:param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
:param input: theano variable to serve as input; if None, a new variable is created
:param output: theano variable to serve as output; if None, a new variable is created
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, n_inputs, n_outputs, n_hiddens, act_fun, n_mades, batch_norm=True, output_order='sequential', mode='sequential', input=None, output=None, rng=np.random):
	    
	
	    # save input arguments
	    self.n_inputs = n_inputs
	    self.n_outputs = n_outputs
	    self.n_hiddens = n_hiddens
	    self.act_fun = act_fun
	    self.n_mades = n_mades
	    self.batch_norm = batch_norm
	    self.mode = mode
	
	    self.input = tt.matrix('x', dtype=dtype) if input is None else input
	    self.y = tt.matrix('y', dtype=dtype) if output is None else output
	    self.parms = []
	
	    self.mades = []
	    self.bns = []
	    self.u = self.y
	    self.logdet_dudy = 0.0
	
	    for i in range(n_mades):
	
	        # create a new made
	        made = mades.ConditionalGaussianMade(n_inputs, n_outputs, n_hiddens, act_fun, output_order, mode, self.input, self.u, rng)
	        self.mades.append(made)
	        self.parms += made.parms
	        output_order = output_order if output_order == 'random' else made.output_order[::-1]
	
	        # inverse autoregressive transform
	        self.u = made.u
	        self.logdet_dudy += 0.5 * tt.sum(made.logp, axis=1)
	
	        # batch normalization
	        if batch_norm:
	            bn = layers.BatchNorm(self.u, n_outputs)
	            self.u = bn.y
	            self.parms += bn.parms
	            self.logdet_dudy += tt.sum(bn.log_gamma) - 0.5 * tt.sum(tt.log(bn.v))
	            self.bns.append(bn)
	
	    self.output_order = self.mades[0].output_order
	
	    # log likelihoods
	    self.L = -0.5 * n_outputs * np.log(2 * np.pi) - 0.5 * tt.sum(self.u ** 2, axis=1) + self.logdet_dudy
	    self.L.name = 'L'
	
	    # train objective
	    self.trn_loss = -tt.mean(self.L)
	    self.trn_loss.name = 'trn_loss'
	
	    # theano evaluation functions, will be compiled when first needed
	    self.eval_lprob_f = None
	    self.eval_grad_f = None
	    self.eval_score_f = None
	    self.eval_us_f = None
	
	```
### *ConditionalMaskedAutoregressiveFlow*.**calc\_random\_numbers**`#!py3 (self, xy)` { #calc\_random\_numbers data-toc-label=calc\_random\_numbers }


```
Given a dataset, calculate the random numbers used internally to generate the dataset.
:param xy: a pair (x, y) of numpy arrays, where x rows are inputs and y rows are outputs
:return: numpy array, rows are corresponding random numbers
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def calc_random_numbers(self, xy):
	    
	
	    # compile theano function, if haven't already done so
	    if self.eval_us_f is None:
	        self.eval_us_f = theano.function(
	            inputs=[self.input, self.y],
	            outputs=self.u
	        )
	
	    x, y, one_datapoint = util.misc.prepare_cond_input(xy, dtype)
	    u = self.eval_us_f(x, y)
	
	    return u[0] if one_datapoint else u
	
	```
### *ConditionalMaskedAutoregressiveFlow*.**eval**`#!py3 (self, xy, log=True)` { #eval data-toc-label=eval }


```
Evaluate log probabilities for given input-output pairs.
:param xy: a pair (x, y) where x rows are inputs and y rows are outputs
:param log: whether to return probabilities in the log domain
:return: log probabilities: log p(y|x)
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def eval(self, xy, log=True):
	    
	
	    # compile theano function, if haven't already done so
	    if self.eval_lprob_f is None:
	        self.eval_lprob_f = theano.function(
	            inputs=[self.input, self.y],
	            outputs=self.L,
	            givens=[(bn.m, bn.bm) for bn in self.bns] + [(bn.v, bn.bv) for bn in self.bns]
	        )
	
	    x, y, one_datapoint = util.misc.prepare_cond_input(xy, dtype)
	
	    lprob = self.eval_lprob_f(x, y)
	    lprob = lprob[0] if one_datapoint else lprob
	
	    return lprob if log else np.exp(lprob)
	
	```
### *ConditionalMaskedAutoregressiveFlow*.**gen**`#!py3 (self, x, n_samples=None, u=None, rng=<module 'numpy.random' from '/Users/jm/opt/anaconda3/envs/ind/lib/python3.7/site-packages/numpy/random/__init__.py'>)` { #gen data-toc-label=gen }


```
Generate samples, by propagating random numbers through each made, after conditioning on input x.
:param x: input vector
:param n_samples: number of samples, 1 if None
:param u: random numbers to use in generating samples; if None, new random numbers are drawn
:return: samples
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen(self, x, n_samples=None, u=None, rng=np.random):
	    
	
	    if n_samples is None:
	        return self.gen(x, 1, u if u is None else u[np.newaxis, :], rng)[0]
	
	    y = rng.randn(n_samples, self.n_outputs).astype(dtype) if u is None else u
	
	    if getattr(self, 'batch_norm', False):
	
	        for made, bn in zip(self.mades[::-1], self.bns[::-1]):
	            y = bn.eval_inv(y)
	            y = made.gen(x, n_samples, y, rng)
	
	    else:
	
	        for made in self.mades[::-1]:
	            y = made.gen(x, n_samples, y, rng)
	
	    return y
	
	```
### *ConditionalMaskedAutoregressiveFlow*.**grad\_log\_p**`#!py3 (self, xy)` { #grad\_log\_p data-toc-label=grad\_log\_p }


```
Evaluate the gradient of the log probability wrt the output, for given input-output pairs.
:param xy: a pair (x, y) where x rows are inputs and y rows are outputs
:return: gradient d/dy log p(y|x)
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def grad_log_p(self, xy):
	    
	
	    # compile theano function, if haven't already done so
	    if getattr(self, 'eval_grad_f', None) is None:
	        self.eval_grad_f = theano.function(
	            inputs=[self.input, self.y],
	            outputs=tt.grad(tt.sum(self.L), self.y),
	            givens=[(bn.m, bn.bm) for bn in self.bns] + [(bn.v, bn.bv) for bn in self.bns]
	        )
	
	    x, y, one_datapoint = util.misc.prepare_cond_input(xy, dtype)
	
	    grad = self.eval_grad_f(x, y)
	    grad = grad[0] if one_datapoint else grad
	
	    return grad
	
	```
### *ConditionalMaskedAutoregressiveFlow*.**reset\_theano\_functions**`#!py3 (self)` { #reset\_theano\_functions data-toc-label=reset\_theano\_functions }


```
Resets theano functions, so that they are compiled again when needed.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reset_theano_functions(self):
	    
	
	    self.eval_lprob_f = None
	    self.eval_grad_f = None
	    self.eval_score_f = None
	    self.eval_us_f = None
	
	    for made in self.mades:
	        made.reset_theano_functions()
	
	    for bn in self.bns:
	        bn.reset_theano_functions()
	
	```
### *ConditionalMaskedAutoregressiveFlow*.**score**`#!py3 (self, xy)` { #score data-toc-label=score }


```
Evaluate the gradient of the log probability wrt the input, for given input-output pairs.
:param xy: a pair (x, y) where x rows are inputs and y rows are outputs
:return: gradient d/dx log p(y|x)
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def score(self, xy):
	    
	
	    # compile theano function, if haven't already done so
	    if self.eval_score_f is None:
	        self.eval_score_f = theano.function(
	            inputs=[self.input, self.y],
	            outputs=tt.grad(tt.sum(self.L), self.input),
	            givens=[(bn.m, bn.bm) for bn in self.bns] + [(bn.v, bn.bv) for bn in self.bns]
	        )
	
	    x, y, one_datapoint = util.misc.prepare_cond_input(xy, dtype)
	
	    grads = self.eval_score_f(x, y)
	    grads = grads[0] if one_datapoint else grads
	
	    return grads
	
	```
