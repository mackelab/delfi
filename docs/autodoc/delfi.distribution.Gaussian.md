## **Gaussian**`#!py3 class` { #Gaussian data-toc-label=Gaussian }


### *Gaussian*.**\_\_imul\_\_**`#!py3 (self, other)` { #\_\_imul\_\_ data-toc-label=\_\_imul\_\_ }


```
Incrementally multiply with another Gaussian
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __imul__(self, other):
	    
	    assert isinstance(other, Gaussian)
	
	    res = self * other
	
	    self.m = res.m
	    self.P = res.P
	    self.C = res.C
	    self.S = res.S
	    self.Pm = res.Pm
	    self.logdetP = res.logdetP
	
	    return res
	
	```
### *Gaussian*.**\_\_init\_\_**`#!py3 (self, m=None, P=None, U=None, S=None, Pm=None, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Gaussian distribution

Initialize a gaussian pdf given a valid combination of its parameters.
Valid combinations are: m-P, m-U, m-S, Pm-P, Pm-U, Pm-S

Focus is on efficient multiplication, division and sampling.

Parameters
----------
m : list or np.array, 1d
    Mean
P : list or np.array, 2d
    Precision
U : list or np.array, 2d
    Upper triangular precision factor such that U'U = P
S : list or np.array, 2d
    Covariance
C : list or np.array, 2d
    Upper or lower triangular covariance factor, in any case S = C'C
Pm : list or np.array, 1d
    Precision times mean such that P*m = Pm
seed : int or None
    If provided, random number generator will be seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, m=None, P=None, U=None, S=None, Pm=None, seed=None):
	    
	    assert m is None or np.asarray(m).ndim == 1
	    assert P is None or np.asarray(P).ndim == 2
	    assert U is None or np.asarray(U).ndim == 2
	    assert S is None or np.asarray(S).ndim == 2
	    assert Pm is None or np.asarray(Pm).ndim == 1
	
	    self.__div__ = self.__truediv__
	    self.__idiv__ = self.__itruediv__
	
	    if m is not None:
	        m = np.asarray(m)
	        self.m = m
	        ndim = self.m.size
	
	        if P is not None:
	            P = np.asarray(P)
	            L = np.linalg.cholesky(P)  # P=LL' (lower triag)
	            self.P = P
	            self.C = np.linalg.inv(L)  # C is lower triangular here
	            # S = C'C = L^{-1}^T L^{-1} = (LL^T)^{-1}
	            self.S = np.dot(self.C.T, self.C)
	            self.Pm = np.dot(P, m)
	            self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))
	
	        elif U is not None:
	            U = np.asarray(U)
	            self.P = np.dot(U.T, U)
	            self.C = np.linalg.inv(U.T)  # C is lower triangular here
	            self.S = np.dot(self.C.T, self.C)
	            self.Pm = np.dot(self.P, m)
	            self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))
	
	        elif S is not None:
	            S = np.asarray(S)
	            self.P = np.linalg.inv(S)
	            self.C = np.linalg.cholesky(S).T  # C is upper triangular here
	            self.S = S
	            self.Pm = np.dot(self.P, m)
	            self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))
	
	        else:
	            raise ValueError('Precision information missing')
	
	    elif Pm is not None:
	        Pm = np.asarray(Pm)
	        self.Pm = Pm
	        ndim = self.Pm.size
	
	        if P is not None:
	            P = np.asarray(P)
	            L = np.linalg.cholesky(P)
	            # L = np.linalg.cholesky(P + 0.001*np.identity(P.shape[0]))
	            self.P = P
	            self.C = np.linalg.inv(L)
	            self.S = np.dot(self.C.T, self.C)
	            self.m = np.linalg.solve(P, Pm)
	            self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))
	
	        elif U is not None:
	            U = np.asarray(U)
	            self.P = np.dot(U.T, U)
	            self.C = np.linalg.inv(U.T)
	            self.S = np.dot(self.C.T, self.C)
	            self.m = np.linalg.solve(self.P, Pm)
	            self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))
	
	        elif S is not None:
	            S = np.asarray(S)
	            self.P = np.linalg.inv(S)
	            self.C = np.linalg.cholesky(S).T
	            self.S = S
	            self.m = np.dot(S, Pm)
	            self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))
	
	        else:
	            raise ValueError('Precision information missing')
	
	    else:
	        raise ValueError('Mean information missing')
	
	    super().__init__(ndim, seed=seed)
	
	```
### *Gaussian*.**\_\_ipow\_\_**`#!py3 (self, power)` { #\_\_ipow\_\_ data-toc-label=\_\_ipow\_\_ }


```
Incrementally raise gaussian to a power
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __ipow__(self, power):
	    
	    res = self ** power
	
	    self.m = res.m
	    self.P = res.P
	    self.C = res.C
	    self.S = res.S
	    self.Pm = res.Pm
	    self.logdetP = res.logdetP
	
	    return res
	
	```
### *Gaussian*.**\_\_itruediv\_\_**`#!py3 (self, other)` { #\_\_itruediv\_\_ data-toc-label=\_\_itruediv\_\_ }


```
Incrementally divide by another Gaussian

Note that the resulting Gaussian might be improper.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __itruediv__(self, other):
	    
	    assert isinstance(other, Gaussian)
	
	    res = self / other
	
	    self.m = res.m
	    self.P = res.P
	    self.C = res.C
	    self.S = res.S
	    self.Pm = res.Pm
	    self.logdetP = res.logdetP
	
	    return res
	
	```
### *Gaussian*.**\_\_mul\_\_**`#!py3 (self, other)` { #\_\_mul\_\_ data-toc-label=\_\_mul\_\_ }


```
Multiply with another Gaussian
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __mul__(self, other):
	    
	    assert isinstance(other, Gaussian)
	
	    P = self.P + other.P
	    Pm = self.Pm + other.Pm
	
	    return Gaussian(P=P, Pm=Pm, seed=self.seed)
	
	```
### *Gaussian*.**\_\_pow\_\_**`#!py3 (self, power, modulo=None)` { #\_\_pow\_\_ data-toc-label=\_\_pow\_\_ }


```
Raise Gaussian to a power and get another Gaussian
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __pow__(self, power, modulo=None):
	    
	    P = power * self.P
	    Pm = power * self.Pm
	
	    return Gaussian(P=P, Pm=Pm, seed=self.seed)
	
	```
### *Gaussian*.**\_\_truediv\_\_**`#!py3 (self, other)` { #\_\_truediv\_\_ data-toc-label=\_\_truediv\_\_ }


```
Divide by another Gaussian

Note that the resulting Gaussian might be improper.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __truediv__(self, other):
	    
	    assert isinstance(other, Gaussian)
	
	    P = self.P - other.P
	    Pm = self.Pm - other.Pm
	
	    return Gaussian(P=P, Pm=Pm, seed=self.seed)
	
	```
### *Gaussian*.**convert\_to\_T**`#!py3 (self, dof)` { #convert\_to\_T data-toc-label=convert\_to\_T }


```
Converts Gaussian to Student T

Parameters
----------
dof : int
    Degrees of freedom
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def convert_to_T(self, dof):
	    
	    return StudentsT(self.m, self.S, dof, seed=self.seed)
	
	```
### *Gaussian*.**eval**`#!py3 (self, x, ii=None, log=True)` { #eval data-toc-label=eval }


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
	
	    if ii is None:
	        assert x.shape[1] == self.ndim, "incorrect data dimension"
	        xm = x - self.m
	        lp = -np.sum(np.dot(xm, self.P) * xm, axis=1)
	        lp += self.logdetP - self.ndim * np.log(2.0 * np.pi)
	        lp *= 0.5
	
	    else:
	        ii = np.atleast_1d(ii)
	        m = self.m[ii]
	        S = self.S[ii][:, ii]
	        if m.size == 1:  # single marginal
	            x = x.reshape(-1, m.size)
	        assert x.shape[1] == m.size
	        if np.linalg.matrix_rank(S)==len(S[:,0]):
	            lp = scipy.stats.multivariate_normal.logpdf(x, m, S, allow_singular=True)
	            lp = np.array([lp]) if x.shape[0] == 1 else lp
	        else:
	            raise ValueError('Rank deficiency in covariance matrix')
	
	    res = lp if log else np.exp(lp)
	    return res
	
	```
### *Gaussian*.**gen**`#!py3 (self, n_samples=1)` { #gen data-toc-label=gen }


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
	    z = self.rng.randn(n_samples, self.ndim)
	    samples = np.dot(z, self.C) + self.m
	    return samples
	
	```
### *Gaussian*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *Gaussian*.**kl**`#!py3 (self, other)` { #kl data-toc-label=kl }


```
Calculates the KL divergence from this to another Gaussian

Direction of KL is KL(this | other)
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def kl(self, other):
	    
	    assert isinstance(other, Gaussian)
	    assert self.ndim == other.ndim
	
	    t1 = np.sum(other.P * self.S)
	
	    m = other.m - self.m
	    t2 = np.dot(m, np.dot(other.P, m))
	
	    t3 = self.logdetP - other.logdetP
	
	    t = 0.5 * (t1 + t2 + t3 - self.ndim)
	
	    return t
	
	```
### *Gaussian*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	
	```
### *Gaussian*.**ztrans**`#!py3 (self, mean, std)` { #ztrans data-toc-label=ztrans }


```
Z-transform

Parameters
----------
mean : array
    Mean vector
std : array
    Std vector

Returns
-------
Gaussian distribution
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def ztrans(self, mean, std):
	    
	    m = (self.m - mean) / std
	    S = self.S / np.outer(std, std)
	    return Gaussian(m=m, S=S, seed=self.seed)
	
	```
### *Gaussian*.**ztrans\_inv**`#!py3 (self, mean, std)` { #ztrans\_inv data-toc-label=ztrans\_inv }


```
Z-transform inverse

Parameters
----------
mean : array
    Mean vector
std : array
    Std vector

Returns
-------
Gaussian distribution
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def ztrans_inv(self, mean, std):
	    
	    m = std * self.m + mean
	    S = np.outer(std, std) * self.S
	    return Gaussian(m=m, S=S, seed=self.seed)
	
	```
