## **Uniform**`#!py3 class` { #Uniform data-toc-label=Uniform }


### *Uniform*.**\_\_init\_\_**`#!py3 (self, lower=0.0, upper=1.0, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Uniform distribution

Parameters
----------
lower : list, or np.array, 1d
    Lower bound(s)
upper : list, or np.array, 1d
    Upper bound(s)
seed : int or None
    If provided, random number generator will be seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, lower=0., upper=1., seed=None):
	    
	    self.lower = np.atleast_1d(lower)
	    self.upper = np.atleast_1d(upper)
	
	    assert self.lower.ndim == self.upper.ndim
	    assert self.lower.ndim == 1
	
	    super().__init__(ndim=len(self.lower), seed=seed)
	
	```
### *Uniform*.**eval**`#!py3 (self, x, ii=None, log=True)` { #eval data-toc-label=eval }


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
	    # See BaseDistribution.py for docstring
	    if ii is None:
	        ii = np.arange(self.ndim)
	    else:
	        ii = np.atleast_1d(ii)
	
	    if x.ndim == 1 and ii.size == 1:
	        x = x.reshape(-1, 1)
	    else:
	        x = np.atleast_2d(x)
	
	    assert x.ndim == 2 and ii.ndim == 1
	    assert x.shape[1] == ii.size
	
	    N = x.shape[0]
	
	    p = 1.0 / np.prod(self.upper[ii] - self.lower[ii])
	    p = p * np.ones((N,))  # broadcasting
	    
	    # truncation of density
	    ind = (x > self.lower[ii]) & (x < self.upper[ii])
	    p[np.prod(ind, axis=1)==0] = 0
	
	    if log:
	        if ind.any() == False:
	            raise ValueError('log probability not defined outside of truncation')
	        else:
	            return np.log(p)
	    else:
	        return p
	
	```
### *Uniform*.**gen**`#!py3 (self, n_samples=1)` { #gen data-toc-label=gen }


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
	def gen(self, n_samples=1):
	    # See BaseDistribution.py for docstring
	    ms = self.rng.rand(n_samples, self.ndim) * (self.upper - self.lower) + self.lower
	    return ms
	
	```
### *Uniform*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *Uniform*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	
	```
