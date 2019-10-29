## **Gamma**`#!py3 class` { #Gamma data-toc-label=Gamma }


### *Gamma*.**\_\_init\_\_**`#!py3 (self, alpha=1.0, beta=1.0, offset=0.0, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Univariate (!) Gamma distribution

Parameters
----------
alpha : list, or np.array, 1d
    Shape parameters
beta : list, or np.array, 1d
    inverse scale paramters
seed : int or None
    If provided, random number generator will be seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, alpha=1., beta=1., offset=0., seed=None):
	    
	    super().__init__(ndim=1, seed=seed)
	
	    alpha, beta = np.atleast_1d(alpha), np.atleast_1d(beta)
	    assert alpha.ndim == 1, 'alpha must be a 1-d array'
	    assert alpha.size == beta.size, 'alpha and beta must match in size'
	    assert np.all(alpha > 0.), 'Should be greater than zero.'
	    assert np.all(beta > 0.), 'Should be greater than zero.'
	    self.alpha = alpha
	    self.beta = beta
	    self.offset = offset
	    self._gamma = gamma(a=alpha, scale=1./beta)
	
	```
### *Gamma*.**eval**`#!py3 (self, x, ii=None, log=True)` { #eval data-toc-label=eval }


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
	
	    # x should have a second dim with length 1, not more
	    x = np.atleast_2d(x)
	    assert x.shape[1] == 1, 'x needs second dim'
	    assert not x.ndim > 2, 'no more than 2 dims in x'
	
	    res = self._gamma.logpdf(x-self.offset) if log else self._gamma.pdf(x-self.offset)
	    # reshape to (nbatch, )
	    return res.reshape(-1)
	
	```
### *Gamma*.**gen**`#!py3 (self, n_samples=1, seed=None)` { #gen data-toc-label=gen }


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
	
	    x = self.rng.gamma(shape=self.alpha,
	                       scale=1./self.beta,
	                       size=(n_samples, self.ndim)) + self.offset
	    return x
	
	```
### *Gamma*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *Gamma*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	
	```
