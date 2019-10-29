## **MoG**`#!py3 class` { #MoG data-toc-label=MoG }


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

### *MoG*.**\_\_imul\_\_**`#!py3 (self, other)` { #\_\_imul\_\_ data-toc-label=\_\_imul\_\_ }


```
Incrementally multiply with a single gaussian
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __imul__(self, other):
	    
	    assert isinstance(other, Gaussian)
	
	    res = self * other
	
	    self.a = res.a
	    self.xs = res.xs
	
	    return res
	
	```
### *MoG*.**\_\_init\_\_**`#!py3 (self, a, ms=None, Ps=None, Us=None, Ss=None, xs=None, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Mixture of Gaussians

Creates a MoG with a valid combination of parameters or an already given
list of Gaussian variables.

Parameters
----------
a : list or np.array, 1d
    Mixing coefficients
ms : list, length n_components
    Means
Ps : list, length n_components
    Precisions
Us : list, length n_components
    Precision factors such that U'U = P
Ss : list, length n_components
    Covariances
xs : list, length n_components
    List of gaussian variables
seed : int or None
    If provided, random number generator will be seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(
	        self,
	        a,
	        ms=None,
	        Ps=None,
	        Us=None,
	        Ss=None,
	        xs=None,
	        seed=None):
	    
	    self.__div__ = self.__truediv__
	    self.__idiv__ = self.__itruediv__
	
	    if ms is not None:
	        super().__init__(
	            a=np.asarray(a),
	            ncomp=len(ms),
	            ndim=np.asarray(
	                ms[0]).ndim,
	            seed=seed)
	
	        if Ps is not None:
	            self.xs = [
	                Gaussian(
	                    m=m, P=P, seed=self.gen_newseed()) for m, P in zip(
	                    ms, Ps)]
	
	        elif Us is not None:
	            self.xs = [
	                Gaussian(
	                    m=m, U=U, seed=self.gen_newseed()) for m, U in zip(
	                    ms, Us)]
	
	        elif Ss is not None:
	            self.xs = [
	                Gaussian(
	                    m=m, S=S, seed=self.gen_newseed()) for m, S in zip(
	                    ms, Ss)]
	
	        else:
	            raise ValueError('Precision information missing')
	
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
### *MoG*.**\_\_itruediv\_\_**`#!py3 (self, other)` { #\_\_itruediv\_\_ data-toc-label=\_\_itruediv\_\_ }


```
Incrementally divide by a single gaussian
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __itruediv__(self, other):
	    
	    assert isinstance(other, Gaussian)
	
	    res = self / other
	
	    self.a = res.a
	    self.xs = res.xs
	
	    return res
	
	```
### *MoG*.**\_\_mul\_\_**`#!py3 (self, other)` { #\_\_mul\_\_ data-toc-label=\_\_mul\_\_ }


```
Multiply with a single gaussian
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __mul__(self, other):
	    
	    assert isinstance(other, Gaussian)
	
	    ys = [x * other for x in self.xs]
	
	    lcs = np.empty_like(self.a)
	
	    for i, (x, y) in enumerate(zip(self.xs, ys)):
	
	        lcs[i] = x.logdetP + other.logdetP - y.logdetP
	        lcs[i] -= np.dot(x.m, np.dot(x.P, x.m)) + \
	            np.dot(other.m, np.dot(other.P, other.m)) - \
	            np.dot(y.m, np.dot(y.P, y.m))
	        lcs[i] *= 0.5
	
	    la = np.log(self.a) + lcs
	    la -= scipy.special.logsumexp(la)
	    a = np.exp(la)
	
	    return MoG(a=a, xs=ys, seed=self.seed)
	
	```
### *MoG*.**\_\_truediv\_\_**`#!py3 (self, other)` { #\_\_truediv\_\_ data-toc-label=\_\_truediv\_\_ }


```
Divide by a single gaussian
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __truediv__(self, other):
	    
	    assert isinstance(other, Gaussian)
	
	    ys = [x / other for x in self.xs]
	
	    lcs = np.empty_like(self.a)
	
	    for i, (x, y) in enumerate(zip(self.xs, ys)):
	
	        lcs[i] = x.logdetP - other.logdetP - y.logdetP
	        lcs[i] -= np.dot(x.m, np.dot(x.P, x.m)) - \
	            np.dot(other.m, np.dot(other.P, other.m)) - \
	            np.dot(y.m, np.dot(y.P, y.m))
	        lcs[i] *= 0.5
	
	    la = np.log(self.a) + lcs
	    la -= scipy.special.logsumexp(la)
	    a = np.exp(la)
	
	    return MoG(a=a, xs=ys, seed=self.seed)
	
	```
### *MoG*.**calc\_mean\_and\_cov**`#!py3 (self)` { #calc\_mean\_and\_cov data-toc-label=calc\_mean\_and\_cov }


```
Calculate the mean vector and the covariance matrix of the MoG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def calc_mean_and_cov(self):
	    
	    ms = [x.m for x in self.xs]
	    m = np.dot(self.a, np.array(ms))
	
	    msqs = [x.S + np.outer(mi, mi) for x, mi in zip(self.xs, ms)]
	    S = np.sum(
	        np.array([a * msq for a, msq in zip(self.a, msqs)]), axis=0) - np.outer(m, m)
	
	    return m, S
	
	```
### *MoG*.**convert\_to\_E**`#!py3 (self, beta=0.99)` { #convert\_to\_E data-toc-label=convert\_to\_E }


```
Convert to Mixture of ellipsoidal distributions
        
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def convert_to_E(self, beta=0.99):
	    
	    return MoE(self.a, xs=self.xs, seed=self.seed, beta=beta)
	
	```
### *MoG*.**convert\_to\_T**`#!py3 (self, dofs)` { #convert\_to\_T data-toc-label=convert\_to\_T }


```
Convert to Mixture of Student's T distributions

Parameters
----------
dofs : int or list of ints
    Degrees of freedom of component distributions
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def convert_to_T(self, dofs):
	    
	    if type(dofs) == int:
	        dofs = [dofs for i in range(len(self.xs))]
	    ys = [x.convert_to_T(dof) for x, dof in zip(self.xs, dofs)]
	    return MoT(self.a, xs=ys, seed=self.seed)
	
	```
### *MoG*.**eval**`#!py3 (self, x, ii=None, log=True)` { #eval data-toc-label=eval }


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
	    ps = np.array([c.eval(x, ii, log) for c in self.xs]).T
	    res = scipy.special.logsumexp(
	        ps +
	        np.log(
	            self.a),
	        axis=1) if log else np.dot(
	        ps,
	        self.a)
	
	    return res
	
	```
### *MoG*.**gen**`#!py3 (self, n_samples=1)` { #gen data-toc-label=gen }


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
### *MoG*.**gen\_comp**`#!py3 (self, n_samples)` { #gen\_comp data-toc-label=gen\_comp }


```
Generate component index according to self.a
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen_comp(self, n_samples):
	    
	    return self.discrete_sample.gen(n_samples).reshape(-1)  # n_samples,
	
	```
### *MoG*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *MoG*.**kl**`#!py3 (self, other, n_samples=10000)` { #kl data-toc-label=kl }


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
### *MoG*.**project\_to\_gaussian**`#!py3 (self)` { #project\_to\_gaussian data-toc-label=project\_to\_gaussian }


```
Returns a gaussian with the same mean and precision as the mog
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def project_to_gaussian(self):
	    
	    m, S = self.calc_mean_and_cov()
	    return Gaussian(m=m, S=S, seed=self.seed)
	
	```
### *MoG*.**prune\_negligible\_components**`#!py3 (self, threshold)` { #prune\_negligible\_components data-toc-label=prune\_negligible\_components }


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
### *MoG*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


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
### *MoG*.**ztrans**`#!py3 (self, mean, std)` { #ztrans data-toc-label=ztrans }


```
Z-transform
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def ztrans(self, mean, std):
	    
	    xs = [x.ztrans(mean, std) for x in self.xs]
	    return MoG(self.a, xs=xs, seed=self.seed)
	
	```
### *MoG*.**ztrans\_inv**`#!py3 (self, mean, std)` { #ztrans\_inv data-toc-label=ztrans\_inv }


```
Z-transform inverse
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def ztrans_inv(self, mean, std):
	    
	    xs = [x.ztrans_inv(mean, std) for x in self.xs]
	    return MoG(self.a, xs=xs, seed=self.seed)
	
	```
