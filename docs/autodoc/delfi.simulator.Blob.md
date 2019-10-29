## **Blob**`#!py3 class` { #Blob data-toc-label=Blob }


### *Blob*.**\_\_init\_\_**`#!py3 (self, isize=32, maxval=255, sigma=None, seed=None, xy_abs_max=17, gamma_min=0.2, gamma_max=5.05)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Gauss simulator

Toy model that generates images containing a blob. For details, see
figure 3 of https://arxiv.org/pdf/1805.09294.pdf

Parameters
----------
isize: int
    Number of image rows and columns
maxval: int
    Maximum pixel value
xy_abs_max: int
    Maximum distance of blob center from image center, in pixels
gamma_min: float
    Parameter controlling blob shape
gamma__max: float
    Parameter controlling blob shape
sigma : float
    Sigma value. If none, it will become a 4th parameter for inference.
seed : int or None
    If set, randomness is seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, isize=32, maxval=255, sigma=None, seed=None,
	             xy_abs_max=17, gamma_min=0.2, gamma_max=5.05):
	    
	    dim = 4 if sigma is None else 3
	    super().__init__(dim_param=dim, seed=seed)
	    self.isize, self.maxval = isize, maxval
	    self.xy_abs_max, self.gamma_min, self.gamma_max = \
	        xy_abs_max, gamma_min, gamma_max
	    self.x, self.y = \
	        np.meshgrid(np.linspace(-isize // 2, isize // 2, isize),
	                    np.linspace(-isize // 2, isize // 2, isize))
	    self.sigma = sigma
	
	```
### *Blob*.**gen**`#!py3 (self, params_list, n_reps=1, pbar=None)` { #gen data-toc-label=gen }


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
### *Blob*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *Blob*.**gen\_single**`#!py3 (self, params)` { #gen\_single data-toc-label=gen\_single }


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
	@copy_ancestor_docstring
	def gen_single(self, params):
	    # See BaseSimulator for docstring
	    if self.sigma is None:
	        assert params.size == 4
	        xo, yo, gamma, sigma = params
	    else:
	        assert params.size == 3
	        xo, yo, gamma = params
	        sigma = self.sigma
	
	    xo = self.xy_abs_max * (2.0 / (1.0 + np.exp(-xo)) - 1.0)
	    yo = self.xy_abs_max * (2.0 / (1.0 + np.exp(-yo)) - 1.0)
	    gamma = (self.gamma_max - self.gamma_min) / (1. + np.exp(-gamma)) \
	        + self.gamma_min
	
	    r = (self.x - xo) ** 2 + (self.y - yo) ** 2
	    p = 0.1 + 0.8 * np.exp(-0.5 * (r / sigma ** 2) ** gamma)
	
	    counts = self.rng.binomial(self.maxval, p) / self.maxval
	
	    return {'data': counts.reshape(-1)}
	
	```
### *Blob*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	
	```
