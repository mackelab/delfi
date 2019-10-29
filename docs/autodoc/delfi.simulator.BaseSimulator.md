## **BaseSimulator**`#!py3 class` { #BaseSimulator data-toc-label=BaseSimulator }


### *BaseSimulator*.**\_\_init\_\_**`#!py3 (self, dim_param, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Abstract base class for simulator models

Simulator models must at least implement abstract methods and properties
of this class.

Parameters
----------
dim_param : int
    Dimensionality of parameter vector
seed : int or None
    If provided, random number generator will be seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, dim_param, seed=None):
	    
	    self.dim_param = dim_param
	
	    self.rng = np.random.RandomState(seed=seed)
	    self.seed = seed
	
	```
### *BaseSimulator*.**gen**`#!py3 (self, params_list, n_reps=1, pbar=None)` { #gen data-toc-label=gen }


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
### *BaseSimulator*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *BaseSimulator*.**gen\_single**`#!py3 (self, params)` { #gen\_single data-toc-label=gen\_single }


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
	@abc.abstractmethod
	def gen_single(self, params):
	    
	    pass
	
	```
### *BaseSimulator*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	
	```
