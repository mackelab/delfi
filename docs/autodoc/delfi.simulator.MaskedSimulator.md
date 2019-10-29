## **MaskedSimulator**`#!py3 class` { #MaskedSimulator data-toc-label=MaskedSimulator }


### *MaskedSimulator*.**\_\_init\_\_**`#!py3 (self, sim, mask, obs, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Simulator with masked parameters

This is a wrapper around BaseSimulator which imputes
fixed values for specified parameters, reducing the 
dimensionality of the problem.

Parameters
----------
sim : BaseSimulator
    The original simulator
mask : 1d array 
    Boolean array determining the values to be imputed. False corresponds to imputed entries.
obs : 1d array
    Array of parameters from which to impute the values
seed : int or None
    See BaseSimulator
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, sim, mask, obs, seed=None):
	    
	    assert len(mask) == sim.dim_param, "Mask for simulator has incorrect length"
	
	    super().__init__(dim_param=np.count_nonzero(mask), seed=seed)
	    self.sim = sim
	
	    self.mask = mask
	    self.obs = obs
	
	```
### *MaskedSimulator*.**gen**`#!py3 (self, params_list, n_reps=1, pbar=None)` { #gen data-toc-label=gen }


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
### *MaskedSimulator*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


```
Generates a new random seed
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen_newseed(self):
	    
	    if self.seed is None:
	        return None
	    else:
	        return self.rng.randint(0, 2**31)
	
	```
### *MaskedSimulator*.**gen\_single**`#!py3 (self, params)` { #gen\_single data-toc-label=gen\_single }


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
	def gen_single(self, params):
	    real_params = self.obs.copy()
	    real_params[self.mask] = params
	    return self.sim.gen_single(real_params)
	
	```
### *MaskedSimulator*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	
	```
