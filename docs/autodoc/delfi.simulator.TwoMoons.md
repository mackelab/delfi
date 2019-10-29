## **TwoMoons**`#!py3 class` { #TwoMoons data-toc-label=TwoMoons }


### *TwoMoons*.**\_\_init\_\_**`#!py3 (self, mean_radius=1.0, sd_radius=0.1, baseoffset=1.0, mapfunc=None, mapfunc_inverse=None, mapfunc_Jacobian_determinant=None, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Two Moons simulator

Toy model that draws data from a crescent shaped mixture distribution.
For the default mapfunc, this leads to a bimodal posterior, with each
mode the same shape as the simulator's data density.

Parameters
----------
mean_radius: float
    Radius of curvature for each moon in the posterior
sd_radius: float
    Dispersion of samples perpendicular to moon curvature
base_offset: float
    Minimum separation between moons in the posterior
mapfunc: callable or None
    Mapping applied to points. Default as described in Greenberg et al., 2019
mapfunc_inverse: callable or None
    Inverse of mapping
mapfunc_Jacobian_determinant: callable or None
    determinant of Jacobian of manfunc, used for change of variables when calculating likelihood
seed : int or None
    If set, randomness is seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, mean_radius=1.0, sd_radius=0.1, baseoffset=1.0,
	             mapfunc=None, mapfunc_inverse=None, mapfunc_Jacobian_determinant=None,  # transforms noise dist.
	             seed=None):
	    
	    super().__init__(dim_param=2, seed=seed)
	    self.mean_radius = mean_radius
	    self.sd_radius = sd_radius
	    self.baseoffset = baseoffset
	    if mapfunc is None:
	        self.mapfunc = default_mapfunc
	        self.mapfunc_inverse = default_mapfunc_inverse
	        self.mapfunc_Jacobian_determinant = default_mapfunc_Jacobian_determinant
	    else:
	        self.mapfunc, self.mapfunc_inverse, self.mapfunc_Jacobian_determinant = \
	            mapfunc, mapfunc_inverse, mapfunc_Jacobian_determinant
	
	```
### *TwoMoons*.**gen**`#!py3 (self, params_list, n_reps=1, pbar=None)` { #gen data-toc-label=gen }


```
Forward model for simulator for list of parameters

Parameters
----------
params_list : list of lists or 1-d np.arrays
    List of parameter vectors, each of which will be simulated
n_reps : int
    If greater than 1, generate multiple samples given param
pbar : tqdm.tqdm or None
    If None, will do nothing. Otherwise it will call pbar.update(1)
    after each sample.

Returns
-------
data_list : list of lists containing n_reps dicts with data
    Repetitions are runs with the same parameter set, different
    repetitions. Each dictionary must contain a key data that contains
    the results of the forward run. Additional entries can be present.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen(self, params_list, n_reps=1, pbar=None):
	    
	    data_list = []
	    for param in params_list:
	        rep_list = []
	        for r in range(n_reps):
	            rep_list.append(self.gen_single(param))
	        data_list.append(rep_list)
	        if pbar is not None:
	            pbar.update(1)
	
	    return data_list
	
	```
### *TwoMoons*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *TwoMoons*.**gen\_posterior\_samples**`#!py3 (self, obs=array([0., 0.]), prior=None, n_samples=1)` { #gen\_posterior\_samples data-toc-label=gen\_posterior\_samples }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen_posterior_samples(self, obs=np.array([0.0, 0.0]), prior=None, n_samples=1):
	    # works only when we use the default_mapfunc above
	
	    # use opposite rotation as above
	    ang = -np.pi / 4.0
	    c = np.cos(-ang)
	    s = np.sin(-ang)
	
	    theta = np.zeros((n_samples, 2))
	    for i in range(n_samples):
	        p = self.gen_single(np.zeros(2))['data']
	        q = np.zeros(2)
	        q[0] = p[0] - obs[0]
	        q[1] = obs[1] - p[1]
	
	        if np.random.rand() < 0.5:
	            q[0] = -q[0]
	
	        theta[i, 0] = c * q[0] - s * q[1]
	        theta[i, 1] = s * q[0] + c * q[1]
	
	    return theta
	
	```
### *TwoMoons*.**gen\_single**`#!py3 (self, param)` { #gen\_single data-toc-label=gen\_single }


```
Forward model for simulator for single parameter set

Parameters
----------
params : list or np.array, 1d of length dim_param
    Parameter vector

Returns
-------
dict : dictionary with data
    The dictionary must contain a key data that contains the results of
    the forward run. Additional entries can be present.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	@copy_ancestor_docstring
	def gen_single(self, param):
	    # See BaseSimulator for docstring
	    param = np.asarray(param).reshape(-1)
	    assert param.ndim == 1
	    assert param.shape[0] == self.dim_param
	
	    a = np.pi * (self.rng.rand() - 0.5)
	    r = self.mean_radius + self.rng.randn() * self.sd_radius
	    p = np.array([r * np.cos(a) + self.baseoffset, r * np.sin(a)])
	    return {'data': self.mapfunc(param, p)}
	
	```
### *TwoMoons*.**likelihood**`#!py3 (self, param, x, log=True)` { #likelihood data-toc-label=likelihood }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def likelihood(self, param, x, log=True):
	    assert x.size == 2, "not yet implemented for evaluation on multiple points at once"
	    assert np.isfinite(x).all() and (np.imag((x)) == 0).all(), "invalid input"
	    if self.mapfunc_inverse is None or self.mapfunc_Jacobian_determinant is None:
	        return np.nan
	    p = self.mapfunc_inverse(param, x)
	    assert p.size == 2, "not yet implemented for non-bijective map functions"
	    u = p[0] - self.baseoffset
	    v = p[1]
	
	    if u < 0.0:  # invalid x for this theta
	        return -np.inf if log else 0.0
	
	    r = np.sqrt(u ** 2 + v ** 2)  # note the angle distribution is uniform
	    L = -0.5 * ((r - self.mean_radius) / self.sd_radius) ** 2 - 0.5 * np.log(2 * np.pi * self.sd_radius ** 2)
	    return L if log else np.exp(L)
	
	```
### *TwoMoons*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	
	```
