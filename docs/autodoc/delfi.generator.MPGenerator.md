## **MPGenerator**`#!py3 class` { #MPGenerator data-toc-label=MPGenerator }


### *MPGenerator*.**\_\_del\_\_**`#!py3 (self)` { #\_\_del\_\_ data-toc-label=\_\_del\_\_ }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __del__(self):
	    self.stop_workers()
	
	```
### *MPGenerator*.**\_\_init\_\_**`#!py3 (self, models, prior, summary, rej=None, seed=None, verbose=False)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
Generator supporting multiprocessing

Parameters
----------
models : List of simulator instances
    Forward models
prior : Distribution or Mixture instance
    Prior over parameters
summary : SummaryStats instance
    Summary statistics
rej : Function
    Rejection kernel

Attributes
----------
proposal : None or Distribution or Mixture instance
    Proposal prior over parameters. If specified, will generate
    samples given parameters drawn from proposal distribution rather
    than samples drawn from prior when `gen` is called.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, models, prior, summary, rej=None, seed=None, verbose=False):
	    
	    super().__init__(model=None, prior=prior, summary=summary, seed=seed)
	
	    self.rej = rej if rej is not None else default_MPGenerator_rej
	    self.verbose = verbose
	    self.models = models
	
	```
### *MPGenerator*.**\_feedback\_forward\_model**`#!py3 (self, data)` { #\_feedback\_forward\_model data-toc-label=\_feedback\_forward\_model }


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
### *MPGenerator*.**\_feedback\_proposed\_param**`#!py3 (self, param)` { #\_feedback\_proposed\_param data-toc-label=\_feedback\_proposed\_param }


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
### *MPGenerator*.**\_feedback\_summary\_stats**`#!py3 (self, sum_stats)` { #\_feedback\_summary\_stats data-toc-label=\_feedback\_summary\_stats }


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
	def _feedback_summary_stats(self, sum_stats):
	    
	    if self.rej(sum_stats):
	        return 'accept'
	    else:
	        return 'discard'
	
	```
### *MPGenerator*.**draw\_params**`#!py3 (self, n_samples, skip_feedback=False, prior_mixin=0, verbose=True, leave_pbar=True)` { #draw\_params data-toc-label=draw\_params }



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
### *MPGenerator*.**filter\_data**`#!py3 (self, stats, params, skip_feedback=False)` { #filter\_data data-toc-label=filter\_data }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def filter_data(self, stats, params, skip_feedback=False):
	    if skip_feedback == True:
	        return stats, params
	
	    ret_stats = []
	    ret_params = []
	
	    for stat, param in zip(stats, params):
	        response = self._feedback_summary_stats(stat)
	        if response == 'accept':
	            ret_stats.append(stat)
	            ret_params.append(param)
	        elif response == 'discard':
	            continue
	        else:
	            raise ValueError('response not supported')
	
	    return ret_stats, ret_params
	
	```
### *MPGenerator*.**gen**`#!py3 (self, n_samples, n_reps=1, skip_feedback=False, prior_mixin=0, verbose=True, **kwargs)` { #gen data-toc-label=gen }


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
	def gen(self, n_samples, n_reps=1, skip_feedback=False, prior_mixin=0, verbose=True, **kwargs):
	    
	    assert n_reps == 1, 'n_reps > 1 is not yet supported'
	
	    params = self.draw_params(n_samples=n_samples,
	                              skip_feedback=skip_feedback,
	                              prior_mixin=prior_mixin,
	                              verbose=verbose)
	
	    return self.run_model(params, skip_feedback=skip_feedback, verbose=verbose, **kwargs)
	
	```
### *MPGenerator*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *MPGenerator*.**iterate\_minibatches**`#!py3 (self, params, minibatch=50)` { #iterate\_minibatches data-toc-label=iterate\_minibatches }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def iterate_minibatches(self, params, minibatch=50):
	    n_samples = len(params)
	
	    for i in range(0, n_samples - minibatch + 1, minibatch):
	        yield params[i:i + minibatch]
	
	    rem_i = n_samples - (n_samples % minibatch)
	    if rem_i != n_samples:
	        yield params[rem_i:]
	
	```
### *MPGenerator*.**log**`#!py3 (self, msg)` { #log data-toc-label=log }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def log(self, msg):
	    if self.verbose:
	        print("Parent: {}".format(msg))
	
	```
### *MPGenerator*.**process\_batch**`#!py3 (self, params_batch, result, skip_feedback=False)` { #process\_batch data-toc-label=process\_batch }



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
### *MPGenerator*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


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
	    for m in self.models:
	        m.reseed(self.gen_newseed())
	    if self.proposal is not None:
	        self.proposal.reseed(self.gen_newseed())
	
	```
### *MPGenerator*.**run\_model**`#!py3 (self, params, minibatch=50, skip_feedback=False, keep_data=True, verbose=False)` { #run\_model data-toc-label=run\_model }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def run_model(self, params, minibatch=50, skip_feedback=False, keep_data=True, verbose=False):
	    # Run forward model for params (in batches)
	    if not verbose:
	        pbar = no_tqdm()
	    else:
	        pbar = progressbar(total=len(params))
	        desc = 'Run simulations '
	        if type(verbose) == str:
	            desc += verbose
	        pbar.set_description(desc)
	
	    self.start_workers()
	    final_params = []
	    final_stats = []  # list of summary stats
	    minibatches = self.iterate_minibatches(params, minibatch)
	    done = False
	    with pbar:
	        while not done:
	            active_list = []
	            for w, p in zip(self.workers, self.pipes):
	                try:
	                    params_batch = next(minibatches)
	                except StopIteration:
	                    done = True
	                    break
	
	                active_list.append((w, p))
	                self.log("Dispatching to worker (len = {})".format(len(params_batch)))
	                p.send(params_batch)
	                self.log("Done")
	
	            n_remaining = len(active_list)
	            while n_remaining > 0:
	                self.log("Listening to worker")
	                msg = self.queue.get()
	                if type(msg) == int:
	                    self.log("Received int")
	                    pbar.update(msg)
	                elif type(msg) == tuple:
	                    self.log("Received results")
	                    stats, params = self.filter_data(*msg, skip_feedback=skip_feedback)
	                    final_stats += stats
	                    final_params += params
	                    n_remaining -= 1
	                else:
	                    self.log("Warning: Received unknown message of type {}".format(type(msg)))
	
	    self.stop_workers()
	
	    # TODO: for n_reps > 1 duplicate params; reshape stats array
	
	    # n_samples x n_reps x dim theta
	    params = np.array(final_params)
	
	    # n_samples x n_reps x dim summary stats
	    stats = np.array(final_stats)
	    stats = stats.squeeze(axis=1)
	
	    return params, stats
	
	```
### *MPGenerator*.**start\_workers**`#!py3 (self)` { #start\_workers data-toc-label=start\_workers }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def start_workers(self):
	    pipes = [ mp.Pipe(duplex=True) for m in self.models ]
	    self.queue = mp.Queue()
	    self.workers = [ Worker(i, self.queue, pipes[i][1], self.models[i], self.summary, seed=self.rng.randint(low=0,high=2**31), verbose=self.verbose) for i in range(len(self.models)) ]
	    self.pipes = [ p[0] for p in pipes ]
	
	    self.log("Starting workers")
	    for w in self.workers:
	        w.start()
	
	    self.log("Done")
	
	```
### *MPGenerator*.**stop\_workers**`#!py3 (self)` { #stop\_workers data-toc-label=stop\_workers }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def stop_workers(self):
	    if self.workers is None:
	        return
	
	    self.log("Closing")
	    for w, p in zip(self.workers, self.pipes):
	        self.log("Closing pipe")
	        p.close()
	
	    for w in self.workers:
	        self.log("Joining process")
	        w.join(timeout=1)
	        w.terminate()
	
	    self.queue.close()
	
	    self.workers = None
	    self.pipes = None
	    self.queue = None
	
	```
