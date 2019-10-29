## **TransformedNormal**`#!py3 class` { #TransformedNormal data-toc-label=TransformedNormal }


### *TransformedNormal*.**\_\_init\_\_**`#!py3 (self, m=None, P=None, U=None, S=None, Pm=None, upper=None, lower=None, flags=None, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
multivariate normals with some entries in log- and/or logit-space.

Initialize the pdf given a valid combination of its parameters.
Valid combinations are: m-P, m-U, m-S, Pm-P, Pm-U, Pm-S

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
flags: list or np.array, 1d
    List of flags for each variable whether it is untransformed (=0),
    log-transformed (=1) or logit-transformed (=2).
lower: list or np.array, 1d
    Lower bounds for logit-box. Defaults to 0 for each parameter.
upper: list or np.array, 1d
    Upper bounds for logit-box. Defaults to 1 for each parameter.
seed : int or None
    If provided, random number generator will be seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, m=None, P=None, U=None, S=None, Pm=None, 
	             upper=None, lower=None, flags=None, seed=None):
	    
	
	    assert m is None or np.asarray(m).ndim == 1
	    assert P is None or np.asarray(P).ndim == 2
	    assert U is None or np.asarray(U).ndim == 2
	    assert S is None or np.asarray(S).ndim == 2
	    assert Pm is None or np.asarray(Pm).ndim == 1
	
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
	
	    self.lower = np.zeros_like(m) if lower is None else np.atleast_1d(lower)
	    self.upper = np.ones_like(m)  if upper is None else np.atleast_1d(upper)
	
	    assert self.lower.ndim == self.upper.ndim
	    assert self.lower.ndim == 1            
	
	    self.flags = np.zeros_like(m) if flags is None else np.atleast_1d(flags)
	
	    assert self.flags.ndim == 1            
	    assert np.all(np.in1d(np.unique(self.flags), np.arange(3)))
	
	    super().__init__(ndim, seed=seed)
	
	```
### *TransformedNormal*.**\_f**`#!py3 (self, x)` { #\_f data-toc-label=\_f }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def _f(self, x):
	
	    y = x.copy()
	
	    # log-transformed entries
	    idx = np.where(self.flags==1.)[0]
	    y[:,idx] = np.exp(y[:,idx])
	
	    # logit-transformed entries
	    idx = np.where(self.flags==2.)[0]
	    y[:,idx] = (self.upper[idx] - self.lower[idx]) / (1. + np.exp( - y[:,idx]) ) + self.lower[idx]
	
	    return y
	
	```
### *TransformedNormal*.**\_finv**`#!py3 (self, y, ii)` { #\_finv data-toc-label=\_finv }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def _finv(self, y, ii):
	
	    x = y.copy()
	
	    #ii = np.arange(self.ndim) if ii is None else ii
	
	    ii_ = np.intersect1d(np.where(self.flags==1.)[0], ii)
	    idx = np.where(np.in1d(ii, ii_))[0]
	    if len(idx) > 0:
	        x[:,idx] = np.log(x[:,idx])
	
	    # logit-transformed entries
	    ii_ = np.intersect1d(np.where(self.flags==2.)[0], ii)
	    idx = np.where(np.in1d(ii, ii_))[0]
	    if len(idx) > 0:
	        x[:,idx] = (x[:,idx] - self.lower[ii_]) / (self.upper[ii_] - self.lower[ii_]) 
	        x[:,idx] = np.log(x[:,idx]) - np.log(1. - x[:,idx])
	
	    return x
	
	```
### *TransformedNormal*.**\_logZ**`#!py3 (self, x, ii)` { #\_logZ data-toc-label=\_logZ }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def _logZ(self, x, ii):
	
	    logZ = 0.
	
	    # log-transformed entries
	    ii_ = np.intersect1d(np.where(self.flags==1.)[0], ii)
	    idx = np.where(np.in1d(ii, ii_))[0]
	    if len(idx) > 0:
	        logZ += np.sum(np.log(x[:,idx]), axis=1)
	
	    # logit-transformed entries
	    ii_ = np.intersect1d(np.where(self.flags==2.)[0], ii)
	    idx = np.where(np.in1d(ii, ii_))[0]
	    if len(idx) > 0:
	        x_ = (x[:,idx] - self.lower[ii_]) / (self.upper[ii_] - self.lower[ii_])
	        logZ += np.sum(np.log(x_ * (1. - x_)), axis=1)
	        logZ += np.sum(np.log(self.upper[ii_]-self.lower[ii_])) # account for overall volume
	
	    return logZ
	
	```
### *TransformedNormal*.**eval**`#!py3 (self, x, ii=None, log=True)` { #eval data-toc-label=eval }


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
	
	    if x.ndim==1:
	        x = x.reshape(-1,1)
	
	
	    if ii is None:
	
	        finv = self._finv(x, np.arange(self.ndim))
	        xm = finv - self.m
	        lp = -np.sum(np.dot(xm, self.P) * xm, axis=1)
	        lp += self.logdetP - self.ndim * np.log(2.0 * np.pi) 
	        lp *= 0.5
	        lp -= self._logZ(x, np.arange(self.ndim))
	
	    else:
	
	        finv = self._finv(x, ii)
	        m = self.m[ii]
	        S = self.S[ii][:, ii]
	
	        if np.linalg.matrix_rank(S)==len(S[:,0]):
	            lp = scipy.stats.multivariate_normal.logpdf(finv, m, S, allow_singular=True)
	            lp = np.array([lp]) if x.shape[0] == 1 else lp
	            lp -= self._logZ(x, ii)
	        else:
	            raise ValueError('Rank deficiency in covariance matrix')
	
	    res = lp if log else np.exp(lp)
	    return res
	
	```
### *TransformedNormal*.**gen**`#!py3 (self, n_samples=1)` { #gen data-toc-label=gen }


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
	    return self._f(samples)
	
	```
### *TransformedNormal*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *TransformedNormal*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	
	```
