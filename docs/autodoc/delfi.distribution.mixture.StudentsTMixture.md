## **MoT**`#!py3 class` { #MoT data-toc-label=MoT }


```
Abstract base class for mixture distributions

Distributions must at least implement abstract methods of this class.

Component distributions should be added to self.xs, which is a list
containing the distributions of individual components.

Parameters
----------
a : list or np.array, 1d
    Mixing coefficients
ncomp : int
    Number of components
ndim : int
    Number of ndimensions of the component distributions
seed : int or None
    If provided, random number generator will be seeded
```

### *MoT*.**\_\_init\_\_**`#!py3 (self, a, ms=None, Ss=None, dofs=None, xs=None, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Mixture of Student's T distributions

Creates a MoT with a valid combination of parameters or an already given
list of gaussian variables.

Parameters
----------
a : list or 1d array
    Mixing coefficients
ms : list of length n_components
    Means
Ss : list of length n_components
    Covariances
dofs: list of length n_components
    Degrees of freedom
xs : list of length n_components
    List of Student's T distributions
seed : int or None
    If provided, random number generator will be seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, a, ms=None, Ss=None, dofs=None, xs=None, seed=None):
	    
	    if ms is not None:
	        super().__init__(
	            a=np.asarray(a),
	            ncomp=len(ms),
	            ndim=np.asarray(
	                ms[0]).ndim,
	            seed=seed)
	        self.xs = [
	            StudentsT(
	                m=m,
	                S=S,
	                dof=dof,
	                seed=self.gen_newseed()) for m,
	            S,
	            dof in zip(
	                ms,
	                Ss,
	                dofs)]
	    elif xs is not None:
	        super().__init__(
	            a=np.asarray(a),
	            ncomp=len(xs),
	            ndim=xs[0].ndim,
	            seed=seed)
	        self.xs = xs
	    else:
	        raise ValueError('Mean information missing')
	
	```
### *MoT*.**eval**`#!py3 (self, x, ii=None, log=True)` { #eval data-toc-label=eval }


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
	    # See BaseMixture.py for docstring
	    if ii is not None:
	        raise NotImplementedError
	
	    ps = np.array([c.eval(x, ii=None, log=log) for c in self.xs]).T
	    res = scipy.special.logsumexp(
	        ps +
	        np.log(
	            self.a),
	        axis=1) if log else np.dot(
	        ps,
	        self.a)
	
	    return res
	
	```
### *MoT*.**gen**`#!py3 (self, n_samples=1)` { #gen data-toc-label=gen }


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
	    # See BaseMixture.py for docstring
	    ii = self.gen_comp(n_samples)  # n_samples,
	
	    ns = [np.sum((ii == i).astype(int)) for i in range(self.n_components)]
	    samples = [x.gen(n) for x, n in zip(self.xs, ns)]
	    samples = np.concatenate(samples, axis=0)
	    self.rng.shuffle(samples)
	
	    return samples
	
	```
### *MoT*.**gen\_comp**`#!py3 (self, n_samples)` { #gen\_comp data-toc-label=gen\_comp }


```
Generate component index according to self.a
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen_comp(self, n_samples):
	    
	    return self.discrete_sample.gen(n_samples).reshape(-1)  # n_samples,
	
	```
### *MoT*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *MoT*.**kl**`#!py3 (self, other, n_samples=10000)` { #kl data-toc-label=kl }


```
Estimates the KL from this to another PDF

KL(this | other), using Monte Carlo
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def kl(self, other, n_samples=10000):
	    
	    x = self.gen(n_samples)
	    lp = self.eval(x, log=True)
	    lq = other.eval(x, log=True)
	    t = lp - lq
	
	    res = np.mean(t)
	    err = np.std(t, ddof=1) / np.sqrt(n_samples)
	
	    return res, err
	
	```
### *MoT*.**prune\_negligible\_components**`#!py3 (self, threshold)` { #prune\_negligible\_components data-toc-label=prune\_negligible\_components }


```
Prune components

Removes all the components whose mixing coefficient is less
than a threshold.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def prune_negligible_components(self, threshold):
	    
	    ii = np.nonzero((self.a < threshold).astype(int))[0]
	    total_del_a = np.sum(self.a[ii])
	    del_count = ii.size
	
	    self.ncomp -= del_count
	    self.a = np.delete(self.a, ii)
	    self.a += total_del_a / self.n_components
	    self.xs = [x for i, x in enumerate(self.xs) if i not in ii]
	
	```
### *MoT*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the following RNGs in the following order:
1) Master RNG for the mixture object, using the input seed
2) RNG for the discrete distribution used to sample components. The seed
is generated using the master RNG.
3) RNG for each mixture component, in order. Each seed is generated by
the master RNG.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	    self.discrete_sample.reseed(seed=self.gen_newseed())
	    for x in self.xs:
	        x.reseed(seed=self.gen_newseed())
	
	```
