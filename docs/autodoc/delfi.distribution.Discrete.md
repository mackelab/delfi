## **Discrete**`#!py3 class` { #Discrete data-toc-label=Discrete }


### *Discrete*.**\_\_init\_\_**`#!py3 (self, p, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Discrete distribution

Parameters
----------
p : list or np.array, 1d
    Probabilities of elements, must sum to 1
seed : int or None
    If provided, random number generator will be seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, p, seed=None):
	    
	    super().__init__(ndim=1, seed=seed)
	
	    p = np.asarray(p)
	    assert p.ndim == 1, 'p must be a 1-d array'
	    assert np.isclose(np.sum(p), 1), 'p must sum to 1'
	    self.p = p
	
	```
### *Discrete*.**eval**`#!py3 (self, x, ii=None, log=True)` { #eval data-toc-label=eval }


```
Method to evaluate pdf

Parameters
----------
x : int or list or np.array
    Rows are inputs to evaluate at
ii : list
    A list of indices specifying which marginal to evaluate.
    If None, the joint pdf is evaluated
log : bool, defaulting to True
    If True, the log pdf is evaluated

Returns
-------
scalar
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	@copy_ancestor_docstring
	def eval(self, x, ii=None, log=True):
	    raise NotImplementedError("To be implemented")
	
	```
### *Discrete*.**gen**`#!py3 (self, n_samples=1, seed=None)` { #gen data-toc-label=gen }


```
Method to generate samples

Parameters
----------
n_samples : int
    Number of samples to generate

Returns
-------
n_samples x self.ndim
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	@copy_ancestor_docstring
	def gen(self, n_samples=1, seed=None):
	    # See BaseDistribution.py for docstring
	    c = np.cumsum(self.p[:-1])[np.newaxis, :]  # cdf
	    r = self.rng.rand(n_samples, 1)
	    return np.sum((r > c).astype(int), axis=1).reshape(-1, 1)
	
	```
### *Discrete*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *Discrete*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	
	```
