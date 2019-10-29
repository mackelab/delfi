## **MoGDistractors**`#!py3 class` { #MoGDistractors data-toc-label=MoGDistractors }


### *MoGDistractors*.**\_\_init\_\_**`#!py3 (self, dim=2, noise_cov=1.0, distractors=10, p_true=None, n_samples=1, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Gaussian Mixture simulator

Toy model that draws data from a mixture distribution with 1 "moving" component that depends on the parameters,
and several other "distractor" components that do not.

Parameters
----------
dim : int
    Number of dimensions of parameters
noise_cov : float or dim X dim covariance matrix as array
    Covariance for the moving component
distractors: int or MoG
    MoG components defining the distractors. Will be generated automatically an integer (count) is given.
p_true: float
    Probability that each sampled data point is NOT from a distractor. If None, mixture weights are uniform.
n_samples: int
    Number of data points per simulation, concatenated
seed : int or None
    If set, randomness is seeded
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, dim=2, noise_cov=1.0, distractors=10, p_true=None, n_samples=1, seed=None):
	    
	    super().__init__(dim_param=dim, seed=seed)
	    self.n_samples = n_samples
	    if type(noise_cov) is float:
	        noise_cov = noise_cov * np.eye(dim)
	    self.noise_cov = noise_cov
	    if type(distractors) is int:
	        self.a = np.ones(distractors) / distractors
	        self.ms = np.random.rand(distractors, dim) * 20.0 - 10.0
	        self.Ss = [0.1 + 0.9 * np.diag(np.random.rand(dim)) for _ in range(distractors)]
	    else:
	        assert isinstance(distractors, dd.MoG)
	        self.a = distractors.a
	        self.ms = [x.m for x in distractors.xs]
	        self.Ss = [x.S for x in distractors.xs]
	    if p_true is None:
	        p_true = 1.0 / (self.a.size + 1.0)
	    self.p_true = p_true
	
	```
### *MoGDistractors*.**gen**`#!py3 (self, params_list, n_reps=1, pbar=None)` { #gen data-toc-label=gen }


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
### *MoGDistractors*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *MoGDistractors*.**gen\_single**`#!py3 (self, param)` { #gen\_single data-toc-label=gen\_single }


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
	
	    q_moving = dd.Gaussian(m=param, S=self.noise_cov, seed=self.gen_newseed())
	    q_distractors = dd.MoG(a=self.a, ms=self.ms, Ss=self.Ss, seed=self.gen_newseed())
	
	    samples = []
	    for _ in range(self.n_samples):
	        if np.random.rand() < self.p_true:
	            samples.append(q_moving.gen(1))
	        else:
	            samples.append(q_distractors.gen(1))
	
	    return {'data': np.concatenate(samples, axis=0)}
	
	```
### *MoGDistractors*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Reseeds the distribution's RNG
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	
	```
