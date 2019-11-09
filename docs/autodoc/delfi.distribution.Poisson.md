## **Poisson**`#!py3 class` { #Poisson data-toc-label=Poisson }


### *Poisson*.**\_\_init\_\_**`#!py3 (self, mu=0.0, offset=0.0, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Univariate (!) Poisson distribution
Parameters
----------
mu: shape parameter of the Poisson (Poisson rate)
offset: shift in the mean parameter, see scipy.stats.Poisson documentation. 
seed : int or None
    If provided, random number generator will be seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, mu=0., offset=0., seed=None):
	    
	    super().__init__(ndim=1, seed=seed)
	    
	    mu = np.atleast_1d(mu)
	    assert mu.ndim == 1, 'mu must be a 1-d array'
	    assert offset >= 0, 'offset must not be negative'
	    
	    self.mu = mu
	    self.offset = offset
	    self._poisson = poisson(mu=mu, loc=offset)
	
	```
### *Poisson*.**eval**`#!py3 (self, x, ii=None, log=True)` { #eval data-toc-label=eval }


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
	    # univariate distribution only, i.e. ii=[0] in any case
	    assert ii is None, 'this is a univariate Poisson, ii must be None.'
	
	    # x should have a second dim with length 1, not more
	    x = np.atleast_2d(x)
	    assert x.shape[1] == 1, 'x needs second dim'
	    assert not x.ndim > 2, 'no more than 2 dims in x'
	
	    res = self._poisson.logpmf(x) if log else self._poisson.pmf(x)
	    # reshape to (nbatch, )
	    return res.reshape(-1)
	
	```
### *Poisson*.**gen**`#!py3 (self, n_samples=1, seed=None)` { #gen data-toc-label=gen }


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
	
	    x = self._poisson.rvs(random_state=self.rng, size=(n_samples, self.ndim))
	    return x
	
	```
### *Poisson*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *Poisson*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	
	```
