## **TransformedSimulator**`#!py3 class` { #TransformedSimulator data-toc-label=TransformedSimulator }


### *TransformedSimulator*.**\_\_init\_\_**`#!py3 (self, simulator, inverse_bijection, makecopy=False)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
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
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
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
	
	```
### *TransformedSimulator*.**gen**`#!py3 (self, params_list, n_reps=1, pbar=None)` { #gen data-toc-label=gen }


```
Forward model for simulator for list of parameters

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
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen(self, params_list, n_reps=1, pbar=None):
	    
	    data_list = []
	    for param in params_list:
	        rep_list = []
	        for r in range(n_reps):
	            rep_list.append(self.gen_single(param))
	        data_list.append(rep_list)
	        if pbar is not None:
	            pbar.update(1)
	
	    return data_list
	
	```
### *TransformedSimulator*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


```
Generates a new random seed
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen_newseed(self):
	    return self.simulator.gen_newseed()
	
	```
### *TransformedSimulator*.**gen\_single**`#!py3 (self, input_params)` { #gen\_single data-toc-label=gen\_single }


```
Forward model for simulator for single parameter set

Parameters
----------
params : list or np.array, 1d of length dim_param
    Parameter vector

Returns
-------
dict : dictionary with data
    The dictionary must contain a key data that contains the results of
    the forward run. Additional entries can be present.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen_single(self, input_params):
	    transformed_params = self.inverse_bijection(input_params)
	    return self.simulator.gen_single(transformed_params)
	
	```
### *TransformedSimulator*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    self.simulator.reseed(seed)
	
	```
