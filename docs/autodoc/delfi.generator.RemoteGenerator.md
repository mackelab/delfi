## **RemoteGenerator**`#!py3 class` { #RemoteGenerator data-toc-label=RemoteGenerator }


### *RemoteGenerator*.**\_\_init\_\_**`#!py3 (self, simulator_class, prior, summary_class, hostname, username, simulator_args=None, simulator_kwargs=None, summary_args=None, summary_kwargs=None, save_every=None, remote_python_executable=None, use_slurm=False, slurm_options=None, local_work_path=None, remote_work_path=None, persistent=False, seed=None)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Generator that creates an MPGenerator on a remote server and uses that
to run simulations.

:param persistent: Whether to start a new job if the first one doesn't generate the necessary number of samples.
    This can be useful when the remote host might cancel the job part-way through, combined with save_every.
:param save_every: Progress will be saved on the remote host every time this many samples is generated.
:param simulator_class:
:param prior:
:param summary:
:param hostname:
:param username:
:param remote_python_executable:
:param use_slurm:
:param local_work_path:
:param remote_work_path:
:param seed:
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self,
	             simulator_class, prior, summary_class,
	             hostname, username,
	             simulator_args=None, simulator_kwargs=None, summary_args=None, summary_kwargs=None,
	             save_every=None,
	             remote_python_executable=None, use_slurm=False, slurm_options=None,
	             local_work_path=None, remote_work_path=None, persistent=False,
	             seed=None):
	    
	    super().__init__(model=None, prior=prior, summary=None, seed=seed)
	    self.simulator_class, self.summary_class, self.hostname, self.username,\
	        self.simulator_args, self.simulator_kwargs, self.summary_args, self.summary_kwargs, \
	        self.remote_python_executable, self.local_work_path,\
	        self.remote_work_path, self.use_slurm, self.slurm_options, self.save_every, self.persistent = \
	        simulator_class, summary_class, hostname, username, \
	        simulator_args, simulator_kwargs, summary_args, summary_kwargs, \
	        remote_python_executable, local_work_path, \
	        remote_work_path, use_slurm, slurm_options, save_every, persistent
	    self.time, self.task_time = None, None
	
	```
### *RemoteGenerator*.**\_feedback\_forward\_model**`#!py3 (self, data)` { #\_feedback\_forward\_model data-toc-label=\_feedback\_forward\_model }


```
Feedback step after forward model ran

Parameters
----------
data : np.array
    Data

Returns
-------
response : str
    Supported responses are in ['accept', 'discard']
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	@copy_ancestor_docstring
	def _feedback_forward_model(self, data):
	    # See BaseGenerator for docstring
	    return 'accept'
	
	```
### *RemoteGenerator*.**\_feedback\_proposed\_param**`#!py3 (self, param)` { #\_feedback\_proposed\_param data-toc-label=\_feedback\_proposed\_param }


```
Feedback step after parameter has been proposed

Parameters
----------
param : np.array
    Parameter

Returns
-------
response : str
    Supported responses are in ['accept', 'resample']
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	@copy_ancestor_docstring
	def _feedback_proposed_param(self, param):
	    # See BaseGenerator for docstring
	
	    # if prior is uniform, reject samples outside of bounds
	    # samples might be drawn outside bounds due to proposal
	    if isinstance(self.prior, dd.Uniform):
	        if np.any(param < self.prior.lower) or \
	           np.any(param > self.prior.upper):
	            return 'resample'
	    elif isinstance(self.prior, dd.IndependentJoint):
	        for j, p in enumerate(self.prior.dists):
	            ii = self.prior.dist_index_eachdim == j
	            if isinstance(p, dd.Uniform):
	                if np.any(param[:, ii] < p.lower) or \
	                   np.any(param[:, ii] > p.upper):
	                    return 'resample'
	
	            elif isinstance(p, dd.Gamma):
	                if np.any(param[:, ii] < p.offset):
	                    return 'resample'
	
	    return 'accept'
	
	```
### *RemoteGenerator*.**\_feedback\_summary\_stats**`#!py3 (self, sum_stats)` { #\_feedback\_summary\_stats data-toc-label=\_feedback\_summary\_stats }


```
Feedback step after summary stats were computed

Parameters
----------
sum_stats : np.array
    Summary stats

Returns
-------
response : str
    Supported responses are in ['accept', 'discard']
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	@copy_ancestor_docstring
	def _feedback_summary_stats(self, sum_stats):
	    # See BaseGenerator for docstring
	    return 'accept'
	
	```
### *RemoteGenerator*.**draw\_params**`#!py3 (self, n_samples, skip_feedback=False, prior_mixin=0, verbose=True, leave_pbar=True)` { #draw\_params data-toc-label=draw\_params }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def draw_params(self, n_samples, skip_feedback=False, prior_mixin=0, verbose=True, leave_pbar=True):
	    if not verbose:
	        pbar = no_tqdm()
	    else:
	        pbar = progressbar(total=n_samples, leave=leave_pbar)
	        desc = 'Draw parameters '
	        if type(verbose) == str:
	            desc += verbose
	        pbar.set_description(desc)
	
	    # collect valid parameter vectors from the prior
	    params = []  # list of parameter vectors
	    with pbar:
	        i = 0
	        while i < n_samples:
	            # sample parameter
	            if self.proposal is None or self.rng.random_sample() < prior_mixin:
	                proposed_param = self.prior.gen(n_samples=1)  # dim params,
	            else:
	                proposed_param = self.proposal.gen(n_samples=1)
	
	            # check if parameter vector is valid
	            response = self._feedback_proposed_param(proposed_param)
	            if response == 'accept' or skip_feedback:
	                # add valid param vector to list
	                params.append(proposed_param.reshape(-1))
	                i += 1
	                pbar.update(1)
	            elif response == 'resample':
	                # continue without increment on i or updating the bar
	                continue
	            else:
	                raise ValueError('response not supported')
	
	        return params
	
	```
### *RemoteGenerator*.**gen**`#!py3 (self, n_samples, n_workers=None, persistent=None, **kwargs)` { #gen data-toc-label=gen }


```
Draw parameters and run forward model

Parameters
----------
n_samples : int
    Number of samples
n_reps: int
    Number of repetitions per parameter sample
skip_feedback: bool
    If True, feedback checks on params, data and sum stats are skipped
verbose : bool or str
    If False, will not display progress bars. If a string is passed,
    it will be appended to the description of the progress bar.

Returns
-------
params : n_samples x n_reps x n_params
    Parameters
stats : n_samples x n_reps x n_summary
    Summary statistics of data
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen(self, n_samples, n_workers=None, persistent=None, **kwargs):
	    self.prior.reseed(self.gen_newseed())
	    if self.proposal is not None:
	        self.proposal.reseed(self.gen_newseed())
	
	    if persistent is None:
	        persistent = self.persistent
	
	    samples_remaining, params, stats = n_samples, None, None
	    while samples_remaining > 0:
	        next_params, next_stats, time, task_time = run_remote(self.simulator_class,
	                                                              self.summary_class,
	                                                              self.prior,
	                                                              n_samples,
	                                                              hostname=self.hostname,
	                                                              username=self.username,
	                                                              simulator_args=self.simulator_args,
	                                                              simulator_kwargs=self.simulator_kwargs,
	                                                              summary_args=self.summary_args,
	                                                              summary_kwargs=self.summary_kwargs,
	                                                              remote_python_executable=self.remote_python_executable,
	                                                              remote_work_path=self.remote_work_path,
	                                                              local_work_path=self.local_work_path,
	                                                              proposal=self.proposal,
	                                                              n_workers=n_workers,
	                                                              generator_seed=self.gen_newseed(),
	                                                              use_slurm=self.use_slurm,
	                                                              save_every=self.save_every,
	                                                              slurm_options=self.slurm_options,
	                                                              **kwargs)
	        self.time, self.task_time = time, task_time
	        if params is None:
	            params, stats = next_params, next_stats
	        else:
	            params, stats = np.vstack((params, next_params)), np.vstack((stats, next_stats))
	        samples_remaining -= next_params.shape[0]
	
	        if not persistent:
	            break
	
	    return params, stats
	
	```
### *RemoteGenerator*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *RemoteGenerator*.**iterate\_minibatches**`#!py3 (self, params, minibatch=50)` { #iterate\_minibatches data-toc-label=iterate\_minibatches }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def iterate_minibatches(self, params, minibatch=50):
	    n_samples = len(params)
	
	    for i in range(0, n_samples - minibatch+1, minibatch):
	        yield params[i:i + minibatch]
	
	    rem_i = n_samples - (n_samples % minibatch)
	    if rem_i != n_samples:
	        yield params[rem_i:]
	
	```
### *RemoteGenerator*.**process\_batch**`#!py3 (self, params_batch, result, skip_feedback=False)` { #process\_batch data-toc-label=process\_batch }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def process_batch(self, params_batch, result,skip_feedback=False):
	    ret_stats = []
	    ret_params = []
	
	    # for every datum in data, check validity
	    params_data_valid = []  # list of params with valid data
	    data_valid = []  # list of lists containing n_reps dicts with data
	
	    for param, datum in zip(params_batch, result):
	        # check validity
	        response = self._feedback_forward_model(datum)
	        if response == 'accept' or skip_feedback:
	            data_valid.append(datum)
	            # if data is accepted, accept the param as well
	            params_data_valid.append(param)
	        elif response == 'discard':
	            continue
	        else:
	            raise ValueError('response not supported')
	
	    # for every data in data, calculate summary stats
	    for param, datum in zip(params_data_valid, data_valid):
	        # calculate summary statistics
	        sum_stats = self.summary.calc(datum)  # n_reps x dim stats
	
	        # check validity
	        response = self._feedback_summary_stats(sum_stats)
	        if response == 'accept' or skip_feedback:
	            ret_stats.append(sum_stats)
	            # if sum stats is accepted, accept the param as well
	            ret_params.append(param)
	        elif response == 'discard':
	            continue
	        else:
	            raise ValueError('response not supported')
	
	    return ret_stats, ret_params
	
	```
### *RemoteGenerator*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
Carries out the following operations, in order:
1) Reseeds the master RNG for the generator object, using the input seed
2) Reseeds the prior from the master RNG. This may cause additional
distributions to be reseeded using the prior's RNG (e.g. if the prior is
a mixture)
3) Reseeds the simulator RNG, from the master RNG
4) Reseeds the proposal, if present
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	    self.prior.reseed(self.gen_newseed())
	    if self.model is not None:
	        self.model.reseed(self.gen_newseed())
	    if self.proposal is not None:
	        self.proposal.reseed(self.gen_newseed())
	
	```
