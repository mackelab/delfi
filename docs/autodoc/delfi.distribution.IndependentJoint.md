## **IndependentJoint**`#!py3 class` { #IndependentJoint data-toc-label=IndependentJoint }


```
Joint distribution composed of statistically independent sub-
distributions.

This class defines a concatenation of a list of distributions.
It supports `eval()` and `gen()`.

Parameters
----------
dists : array of distributions
    Array of distributions
seed : int or None
    If provided, random number generator will be seeded
```

### *IndependentJoint*.**\_\_init\_\_**`#!py3 (self, dists, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


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
	def __init__(self, dists, seed=None):
	
	    for d in dists:
	        assert not isinstance(d, IndependentJoint), \
	            "IndependentJoint objects cannot be nested"
	    self.dists = [ d for d in dists if d.ndim > 0 ]
	    self.dimlist = [ d.ndim for d in self.dists]
	    ndim = np.sum(self.dimlist)
	
	    # self.dist_index_eachdim stores the index of the child distribution
	    # for each dimension of the full distribution.
	    self.dist_index_eachdim = np.zeros(ndim, dtype=int)
	    # self.ii_full2child stores the index into the child distribution for
	    # each index into the full distribution
	    self.ii_full2child = np.zeros(ndim, dtype=int)
	
	    # list of indices for the full distribution for each child distribution:
	    self.ii_list_eachdist = []
	    csdims = np.append(0, np.cumsum(self.dimlist))
	    for j in range(len(self.dists)):
	        ii_child = np.arange(csdims[j], csdims[j + 1])
	        self.dist_index_eachdim[ii_child] = j
	        self.ii_full2child[ii_child] = np.arange(self.dimlist[j])
	    super().__init__(ndim=ndim, seed=seed)
	
	```
### *IndependentJoint*.**eval**`#!py3 (self, x, ii=None, log=True)` { #eval data-toc-label=eval }


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
	def eval(self, x, ii=None, log=True):
	    if ii is not None:
	        ii = np.atleast_1d(ii)
	        if ii.dtype == bool:  # convert to array of indices
	            assert ii.size == self.ndim, 'incorrectly sized binary mask'
	            ii = np.flatnozero(ii)
	        assert (np.diff(ii) > 0).all(), 'ii must be increasing'
	        assert (ii >= 0).all() and (ii < self.ndim).all(), 'invalid index'
	
	        xsplit, ds, ii_children = [], [], []
	        for j, d in enumerate(self.dists):
	            # is_childj stores whether each element of ii is in child j
	            is_childj = self.dist_index_eachdim[ii] == j
	            if not is_childj.any():
	                continue
	            ds.append(d)
	            # ii_child stores indices into child distribution's dimensions
	            ii_child = self.ii_full2child[ii[is_childj]]
	            xsplit.append(x[:, is_childj])
	            if ii_child.size == self.dimlist[j]:
	                ii_children.append(None)  # full child distribution
	            else:
	                ii_children.append(ii_child)
	    else:
	        ds = self.dists
	        xsplit = np.split(x, np.cumsum(self.dimlist), axis=-1)
	        ii_children = [None for d in ds]  # full child distribution
	
	    logps = [d.eval(x, ii=ii, log=True)
	             for d, x, ii in zip(ds, xsplit, ii_children)]
	    # each element of logps is a vector with one log prob per data point
	    logp = np.sum(np.vstack(logps), axis=0)
	    return logp if log else np.exp(logp)
	
	```
### *IndependentJoint*.**gen**`#!py3 (self, n_samples=1)` { #gen data-toc-label=gen }


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
	def gen(self, n_samples=1):
	    return np.concatenate([ d.gen(n_samples) for d in self.dists ], axis=-1)
	
	```
### *IndependentJoint*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *IndependentJoint*.**mean**`#!py3 (self)` { #mean data-toc-label=mean }


```
Means
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def mean(self):
	    return np.concatenate([d.mean for d in self.dists])
	
	```
### *IndependentJoint*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    super().reseed(seed)
	    for d in self.dists:
	        d.reseed(self.gen_newseed())
	
	```
### *IndependentJoint*.**std**`#!py3 (self)` { #std data-toc-label=std }


```
Standard deviations of marginals
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def std(self):
	    return np.concatenate([d.std for d in self.dists])
	
	```
