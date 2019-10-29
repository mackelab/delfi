## **MoE**`#!py3 class` { #MoE data-toc-label=MoE }


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

### *MoE*.**\_\_init\_\_**`#!py3 (self, a, ms=None, Ps=None, Us=None, Ss=None, xs=None, seed=None, beta=0.99)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Mixture of Ellipsoidals

Creates a MoE with a valid combination of parameters or an already given
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
beta : float
    Mass to preserve when sampling
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, a, ms=None, Ps=None, Us=None, Ss=None, xs=None,
	             seed=None, beta=0.99):
	    
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
	
	    self.threshold = 2 * gammaincinv(0.5 * self.ndim, beta)
	
	```
### *MoE*.**eval**`#!py3 (self, x, ii=None, log=True)` { #eval data-toc-label=eval }


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
	    # Returns 1 everywhere (unnormalized)
	    if ii is not None:
	        raise NotImplementedError
	
	    ps = np.array([c.eval(x, ii, log) for c in self.xs]).T
	    ps *= 0
	    ps += 1.
	    ps = ps.squeeze()
	
	    if log:
	        return np.log(ps)
	    else:
	        return ps
	
	```
### *MoE*.**gen**`#!py3 (self, n_samples=1)` { #gen data-toc-label=gen }


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
	    for i in range(len(self.xs)):
	        self.xs[i].L = la.cholesky(self.xs[i].S)
	
	    samp = []
	    for _ in range(n_samples):
	        samp.append(self.gen_single())
	    samp = np.array(samp)
	
	    return samp
	
	```
### *MoE*.**gen\_comp**`#!py3 (self, n_samples)` { #gen\_comp data-toc-label=gen\_comp }


```
Generate component index according to self.a
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen_comp(self, n_samples):
	    
	    return self.discrete_sample.gen(n_samples).reshape(-1)  # n_samples,
	
	```
### *MoE*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *MoE*.**gen\_single**`#!py3 (self)` { #gen\_single data-toc-label=gen\_single }


```
Generate single sample
        
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen_single(self):
	    
	    def draw_proposal(xs):
	        μ = xs.m
	        L = xs.L
	        x = self.uni_rand_ellipse(L * np.sqrt(self.threshold))
	        return x.ravel() + μ.ravel()
	
	    while True:
	        i = self.gen_comp(1)[0]  # component index
	        x = draw_proposal(self.xs[i])
	        ρ = np.zeros(self.ncomp)
	        for j, xs in enumerate(self.xs):
	            μ = xs.m
	            L = xs.L
	            z = la.solve(L, (x - μ))
	            ρ[j] = np.dot(z, z)
	        π = 1 / np.sum(ρ < self.threshold)
	        if self.rng.rand() < π:
	            return x
	
	```
### *MoE*.**kl**`#!py3 (self, other, n_samples=10000)` { #kl data-toc-label=kl }


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
### *MoE*.**prune\_negligible\_components**`#!py3 (self, threshold)` { #prune\_negligible\_components data-toc-label=prune\_negligible\_components }


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
### *MoE*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


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
### *MoE*.**uni\_rand\_ellipse**`#!py3 (self, L, n=1)` { #uni\_rand\_ellipse data-toc-label=uni\_rand\_ellipse }


```
Sample from ellipsoid

Parameters
----------
L : np.array
    Cholesky factorization of covariance matrix s.t. Σ = LL'
n : int
    number of samples to generate
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def uni_rand_ellipse(self, L, n=1):
	    
	    m = L.shape[0]
	    x = self.rng.normal(size=(m, n))
	
	    # map the points on the n-dimensional hypersphere
	    w = np.sqrt(np.sum(x ** 2, axis=0))  # norm of the vector
	    x = x / w  # normalized vector
	
	    # generate points distributed as m * r^(m-1) for 0 < r < 1
	    u = self.rng.uniform(size=n)
	    r = np.outer(np.ones(m), u) ** (1 / m)
	
	    φsph = r * x  # φ is uniformely distributed within the unit sphere
	    return np.dot(L, φsph)  # rescale the sphere into an ellipsoid
	
	```
