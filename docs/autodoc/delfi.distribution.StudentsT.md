## **StudentsT**`#!py3 class` { #StudentsT data-toc-label=StudentsT }


### *StudentsT*.**\_\_init\_\_**`#!py3 (self, m, S, dof, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Student's T distribution

Parameters
----------
m : list or np.array, 1d
    Mean
S : list or np.array, 1d
    Covariance
dof : int
    Degrees of freedom
seed : int or None
    If provided, random number generator will be seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, m, S, dof, seed=None):
	    
	    m = np.asarray(m)
	    self.m = m
	    self.dof = dof
	    assert(dof > 0)
	
	    S = np.asarray(S)
	    self.P = np.linalg.inv(S)
	    self.C = np.linalg.cholesky(S).T  # C is upper triangular here
	    self.S = S
	    self.Pm = np.dot(self.P, m)
	    self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))
	    super().__init__(ndim=m.size, seed=seed)
	
	```
### *StudentsT*.**eval**`#!py3 (self, x, ii=None, log=True)` { #eval data-toc-label=eval }


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
	    if ii is not None:
	        raise NotImplementedError
	
	    xm = x - self.m
	    lp = np.log(1 + np.sum(np.dot(xm, self.P) * xm, axis=1) / self.dof)
	    lp *= -(self.dof + self.ndim) / 2.0
	    lp += np.log(scipy.special.gamma((self.dof + self.ndim) / 2))
	    lp -= scipy.special.gammaln(self.dof / 2) + \
	        self.ndim / 2 * np.log(self.dof * np.pi) - 0.5 * self.logdetP
	
	    res = lp if log else np.exp(lp)
	    return res
	
	```
### *StudentsT*.**gen**`#!py3 (self, n_samples=1)` { #gen data-toc-label=gen }


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
	    u = self.rng.chisquare(self.dof, n_samples) / self.dof
	    y = self.rng.multivariate_normal(np.zeros(self.ndim),
	                                      self.S, (n_samples,))
	    return self.m + y / np.sqrt(u)[:, None]
	
	```
### *StudentsT*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *StudentsT*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	
	```
