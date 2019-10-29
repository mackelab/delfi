## **Logistic**`#!py3 class` { #Logistic data-toc-label=Logistic }


### *Logistic*.**\_\_init\_\_**`#!py3 (self, mu=0.0, s=1.0, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Distribution with independent dimensions and logistic marginals

Parameters
---------
mu : list, or np.array, 1d
    Means
s : list, or np.array, 1d
    Scale factors
seed : int or None
    If provided, random number generator will be seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, mu=0.0, s=1.0, seed=None):
	    
	    mu, s = np.atleast_1d(mu), np.atleast_1d(s)
	    if s.size == 1:
	        s = np.full(mu.size, s[0])
	
	    assert (s > 0).all() and np.isfinite(s).all() and np.isfinite(mu).all() and np.isreal(s).all() and \
	           np.isreal(mu).all(), "bad params"
	    assert s.ndim == 1 and mu.ndim == 1 and mu.size == s.size, "bad sizes"
	    self.mu, self.s = mu, s
	
	    super().__init__(ndim=mu.size, seed=seed)
	
	```
### *Logistic*.**eval**`#!py3 (self, x, ii=None, log=True)` { #eval data-toc-label=eval }


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
	    x = np.atleast_2d(x)
	    assert x.shape[1] == self.ndim, "incorrect data dimension"
	
	    if ii is None:
	        ii = np.arange(self.ndim)
	
	    z = (x - self.mu) / self.s
	    logp_eachdim = -z - np.log(self.s) - 2.0 * np.log(1.0 + np.exp(-z))
	    logp = logp_eachdim[:, ii].sum(axis=1)
	
	    return logp if log else np.exp(logp)
	
	```
### *Logistic*.**gen**`#!py3 (self, n_samples=1)` { #gen data-toc-label=gen }


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
	    u = np.random.uniform(size=(n_samples, self.ndim))
	    return self.mu + self.s * (np.log(u) - np.log(1 - u))
	
	```
### *Logistic*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *Logistic*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	
	```
