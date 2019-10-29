## **TransformedDistribution**`#!py3 class` { #TransformedDistribution data-toc-label=TransformedDistribution }


```
Distribution object that carries out an invertible change of variables
for another distribution object.

A bijection must be supplied mapping from
the original distributions variables into the new one's along with the
bijection's inverse and the log determinant of the bijection's Jacobian.

There is no checking of whether the supplied functions are really inverses
of each other or are in fact bijections at all, this is up to the user.

Parameters
----------
distribution : delfi distribution or mixture object
    Original distrib. to be transformed. Must implement eval() and gen()
bijection : callable
    Bijective mapping from original distrib.'s random variable to this one's
inverse_bijection: callable
    Inverse of the bijective mapping, going from this distribution's random
    variable to the original one's.
bijection_jac_logD: callable
    Log determinant of the Jacobian of the bijection from the original distribution's random variable to this one's.
makecopy: bool
    Whether to call deepcopy on the simulator, unlinking the RNGs
```

### *TransformedDistribution*.**\_\_init\_\_**`#!py3 (self, distribution, bijection, inverse_bijection, bijection_jac_logD, makecopy=False)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Abstract base class for distributions

Distributions must at least implement abstract properties and methods of
this class.

Parameters
----------
ndim : int
    Number of ndimensions of the distribution
seed : int or None
    If provided, random number generator will be seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, distribution, bijection, inverse_bijection, bijection_jac_logD, makecopy=False):
	    #assert isinstance(distribution, BaseDistribution) or isinstance(distribution, BaseMixture) \
	    #    or isinstance(distribution, MAFconditional)
	    if makecopy:
	        distribution = deepcopy(distribution)
	    self.distribution = distribution
	    self.bijection, self.inverse_bijection = bijection, inverse_bijection
	    self.bijection_jac_logD = bijection_jac_logD
	    self.ndim = distribution.ndim
	
	```
### *TransformedDistribution*.**eval**`#!py3 (self, x, ii=None, log=True)` { #eval data-toc-label=eval }


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
	    assert ii is None, "cannot marginalize transformed distributions"
	    x_original = self.inverse_bijection(x)
	    logp_original = self.distribution.eval(x_original, log=True)
	    logp = logp_original - self.bijection_jac_logD(x_original)  # change of variables
	    return logp if log else np.exp(logp)
	
	```
### *TransformedDistribution*.**gen**`#!py3 (self, n_samples=1)` { #gen data-toc-label=gen }


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
	    samples = self.distribution.gen(n_samples=n_samples)
	    return self.bijection(samples)
	
	```
### *TransformedDistribution*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


```
Generates a new random seed
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen_newseed(self):
	    
	    return self.distribution.gen_newseed()
	
	```
### *TransformedDistribution*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.distribution.reseed(seed)
	
	```
