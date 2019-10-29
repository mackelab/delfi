## **Identity**`#!py3 class` { #Identity data-toc-label=Identity }


```
Just apply the identity instead of reducing data.
Parameters
----------
idx : list or array of int or bool
    Set of data indices to use us sufficient statistics (None for all).
```

### *Identity*.**\_\_init\_\_**`#!py3 (self, seed=None, idx=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Abstract base class for summary stats

Summary Stats must at least implement abstract methods and properties of
this class: The method ``calc()`` needs to be implemented. The attribute
``n_summary`` can be useful to have, for example to write tests, but it
is not strictly required.

Parameters
----------
seed : int or None
    If provided, random number generator will be seeded

Attributes
----------
n_summary : int
    Number of resulting summary features
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, seed=None, idx=None):
	    super().__init__(seed=seed)
	    self.idx = idx
	
	```
### *Identity*.**calc**`#!py3 (self, repetition_list)` { #calc data-toc-label=calc }


```
Method computing summary statistics

Parameters
----------
repetition_list : list of dictionaries, one per repetition
    data list, returned by `gen` method of Simulator instance

Returns
-------
np.arrray, 2d with n_reps x n_summary
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	@copy_ancestor_docstring
	def calc(self, repetition_list):
	    # See BaseSummaryStats.py for docstring
	
	    # get the number of samples contained
	    n_reps = len(repetition_list)
	
	    # get the size of the data inside a sample
	    self.n_summary = repetition_list[0]['data'].size
	
	    # build a matrix of n_reps x n_summary
	    data_matrix = np.zeros((n_reps, self.n_summary))
	    for rep_idx, rep_dict in enumerate(repetition_list):
	        data_matrix[rep_idx, :] = rep_dict['data']
	
	    return data_matrix if self.idx is None else data_matrix[:, self.idx]
	
	```
### *Identity*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *Identity*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    self.rng.seed(seed=seed)
	    self.seed = seed
	
	```
