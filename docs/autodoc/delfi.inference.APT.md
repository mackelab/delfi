## **SNPEC**`#!py3 class` { #SNPEC data-toc-label=SNPEC }


### *SNPEC*.**\_\_init\_\_**`#!py3 (self, generator, obs=None, prior_norm=False, pilot_samples=100, reg_lambda=0.01, seed=None, verbose=True, add_prior_precision=True, Ptol=None, **kwargs)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
SNPE-C/APT

Implementation of Greenberg, Nonnenmacher & Macke (ICML 2019)

Parameters
----------
generator : generator instance
    Generator instance
obs : array
    Observation in the format the generator returns (1 x n_summary)
prior_norm : bool
    If set to True, will z-transform params based on mean/std of prior
pilot_samples : None or int
    If an integer is provided, a pilot run with the given number of
    samples is run. The mean and std of the summary statistics of the
    pilot samples will be subsequently used to z-transform summary
    statistics.
n_components : int
    Number of components in final round (PM's algorithm 2)
reg_lambda : float
    Precision parameter for weight regularizer if svi is True
seed : int or None
    If provided, random number generator will be seeded
verbose : bool
    Controls whether or not progressbars are shown
add_prior_precision: bool
    Whether to add the prior precision to each posterior component for Gauss/MoG proposals
Ptol: float
    Quantity added to the diagonal entries of the precision matrix for each Gaussian posterior component
kwargs : additional keyword arguments
    Additional arguments for the NeuralNet instance, including:
        n_hiddens : list of ints
            Number of hidden units per layer of the neural network
        svi : bool
            Whether to use SVI version of the network or not

Attributes
----------
observables : dict
    Dictionary containing theano variables that can be monitored while
    training the neural network.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def __init__(self, generator, obs=None, prior_norm=False,
	             pilot_samples=100, reg_lambda=0.01, seed=None, verbose=True,
	             add_prior_precision=True, Ptol=None,
	             **kwargs):
	    
	    assert obs is not None, "APT requires observed data"
	    self.obs = np.asarray(obs)
	    super().__init__(generator, prior_norm=prior_norm,
	                     pilot_samples=pilot_samples, seed=seed,
	                     verbose=verbose, **kwargs)  # initializes network
	    assert 0 < self.obs.ndim <= 2
	    if self.obs.ndim == 1:
	        self.obs = self.obs.reshape(1, -1)
	    assert self.obs.shape[0] == 1
	
	    if np.any(np.isnan(self.obs)):
	        raise ValueError("Observed data contains NaNs")
	
	    self.Ptol = np.finfo(dtype).resolution if Ptol is None else Ptol
	    self.add_prior_precision = add_prior_precision
	    self.reg_lambda = reg_lambda
	    self.exception_info = (None, None, None)
	    self.trn_datasets, self.proposal_used = [], []
	
	```
### *SNPEC*.**centre\_on\_obs**`#!py3 (self)` { #centre\_on\_obs data-toc-label=centre\_on\_obs }


```
Centres first-layer input onto observed summary statistics

Ensures x' = x - xo, i.e. first-layer input x' = 0 for x = xo.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def centre_on_obs(self):
	    
	
	    self.stats_mean = self.obs.copy()
	
	```
### *SNPEC*.**compile\_observables**`#!py3 (self)` { #compile\_observables data-toc-label=compile\_observables }


```
Creates observables dict
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def compile_observables(self):
	    
	    self.observables = {}
	    self.observables['loss.lprobs'] = self.network.lprobs
	    for p in self.network.aps:
	        self.observables[str(p)] = p
	
	```
### *SNPEC*.**conditional\_norm**`#!py3 (self, fcv=0.8, tmu=None, tSig=None, h=None)` { #conditional\_norm data-toc-label=conditional\_norm }


```
Normalizes current network output at observed summary statistics

Parameters
----------
fcv : float
    Fraction of total that comes from uncertainty over components, i.e.
    Var[th] = E[Var[th|z]] + Var[E[th|z]]
            =  (1-fcv)     +     fcv       = 1
tmu: array
    Target mean.
tSig: array
    Target covariance.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def conditional_norm(self, fcv=0.8, tmu=None, tSig=None, h=None):
	    
	
	    # avoid CDELFI.predict() attempt to analytically correct for proposal
	    print('obs', self.obs.shape)
	    print('mean', self.stats_mean.shape)
	    print('std', self.stats_std.shape)
	    obz = (self.obs - self.stats_mean) / self.stats_std
	    posterior = self.network.get_mog(obz.reshape(self.obs.shape),
	                                     deterministic=True)
	    mog = posterior.ztrans_inv(self.params_mean, self.params_std)
	
	    assert np.all(np.diff(mog.a)==0.) # assumes uniform alpha
	
	    n_dim = self.kwargs['n_outputs']
	    triu_mask = np.triu(np.ones([n_dim, n_dim], dtype=dtype), 1)
	    diag_mask = np.eye(n_dim, dtype=dtype)
	
	    # compute MoG mean mu, Sig = E[Var[th|z]] and C = Var[E[th|z]]
	    mu, Sig = np.zeros_like(mog.xs[0].m), np.zeros_like(mog.xs[0].S)
	    for i in range(self.network.n_components):
	        Sig += mog.a[i] * mog.xs[i].S
	        mu  += mog.a[i] * mog.xs[i].m
	    C = np.zeros_like(Sig)
	    for i in range(self.network.n_components):
	        dmu = mog.xs[i].m - mu if self.network.n_components > 1 \
	            else mog.xs[i].m
	        C += mog.a[i] * np.outer(dmu, dmu)
	
	    # if not provided, target zero-mean unit variance (as for prior_norm=True)
	    tmu = np.zeros_like(mog.xs[0].m) if tmu is None else tmu
	    tSig = np.eye(mog.xs[0].m.size) if tSig is None else tSig
	
	    # compute normalizers (we only z-score, don't whiten!)
	    Z1inv = np.sqrt((1.-fcv) / np.diag(Sig) * np.diag(tSig)).reshape(-1)
	    Z2inv = np.sqrt(  fcv    / np.diag( C ) * np.diag(tSig)).reshape(-1)
	
	    # first we need the center of means
	    def idx_MoG(x):
	        return x.name[:5] == 'means'
	    mu_ = np.zeros_like(mog.xs[0].m)
	    for w, b in zip(filter(idx_MoG, self.network.mps_wp),
	                    filter(idx_MoG, self.network.mps_bp)):
	        h = np.zeros(w.get_value().shape[0]) if h is None else h
	        mu_ += h.dot(w.get_value()) + b.get_value()
	    mu_ /= self.network.n_components
	
	    # center and normalize means
	    # mu =  Z2inv * (Wh + b - mu_) + tmu
	    #    = Wh + (Z2inv * (b - mu_ + Wh) - Wh + tum)
	    for w, b in zip(filter(idx_MoG, self.network.mps_wp),
	                    filter(idx_MoG, self.network.mps_bp)):
	        Wh = h.dot(w.get_value())
	        b.set_value(Z2inv * (Wh + b.get_value() - mu_) - Wh + tmu)
	
	    # normalize covariances
	    def idx_MoG(x):
	        return x.name[:10]=='precisions'
	    # Sig^-0.5 = diag_mask * (exp(Wh+b)/exp(log(Z1)) + triu_mask * (Wh+b)*Z1
	    #          = diag_mask *  exp(Wh+ (b-log(Z1))    + triu_mask * (Wh+((b+Wh)*Z1-Wh))
	    for w, b in zip(filter(idx_MoG, self.network.mps_wp),
	                    filter(idx_MoG, self.network.mps_bp)):
	        Wh = h.dot(w.get_value()).reshape(n_dim,n_dim)
	        b_ = b.get_value().copy().reshape(n_dim,n_dim)
	
	        val = diag_mask * (b_ - np.diag(np.log(Z1inv))) + triu_mask * ((b_+Wh).dot(np.diag(1./Z1inv))- Wh )
	
	        b.set_value(val.flatten())
	
	```
### *SNPEC*.**define\_loss**`#!py3 (self, n, round_cl=1, proposal='gaussian', combined_loss=False)` { #define\_loss data-toc-label=define\_loss }


```
Loss function for training

Parameters
----------
n : int
    Number of training samples
round_cl : int
    Round after which to start continual learning
proposal : str
    Specifier for type of proposal used: continuous ('gaussian', 'mog')
    or 'atomic' proposals are implemented.
combined_loss : bool
    Whether to include prior likelihood terms in addition to atomic
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def define_loss(self, n, round_cl=1, proposal='gaussian',
	                combined_loss=False):
	    
	    prior = self.generator.prior
	    if isinstance(prior, dd.Gaussian) or isinstance(prior, dd.MoG):
	        prior = prior.ztrans(self.params_mean, self.params_std)
	
	    if proposal == 'prior':  # using prior as proposal
	        loss, trn_inputs = snpe_loss_prior_as_proposal(self.network, svi=self.svi)
	
	    elif proposal == 'gaussian':
	        assert self.network.density == 'mog'
	        assert isinstance(self.generator.proposal, dd.Gaussian)
	        loss, trn_inputs = apt_loss_gaussian_proposal(self.network, prior, svi=self.svi,
	                                                      add_prior_precision=self.add_prior_precision)
	    elif proposal.lower() == 'mog':
	        assert self.network.density == 'mog'
	        assert isinstance(self.generator.proposal, dd.MoG)
	        loss, trn_inputs = apt_loss_MoG_proposal(self.network, prior, svi=self.svi,
	                                                 add_prior_precision=self.add_prior_precision)
	    elif proposal == 'atomic':
	        loss, trn_inputs = \
	            apt_loss_atomic_proposal(self.network, svi=self.svi, combined_loss=combined_loss)
	    else:
	        raise NotImplemented()
	
	    # adding nodes to dict s.t. they can be monitored during training
	    self.observables['loss.lprobs'] = self.network.lprobs
	    self.observables['loss.raw_loss'] = loss
	
	    if self.svi:
	        if self.round <= round_cl:
	            # weights close to zero-centered prior in the first round
	            if self.reg_lambda > 0:
	                kl, imvs = svi_kl_zero(self.network.mps, self.network.sps,
	                                       self.reg_lambda)
	            else:
	                kl, imvs = 0, {}
	        else:
	            # weights close to those of previous round
	            kl, imvs = svi_kl_init(self.network.mps, self.network.sps)
	
	        loss = loss + 1 / n * kl
	
	        # adding nodes to dict s.t. they can be monitored
	        self.observables['loss.kl'] = kl
	        self.observables.update(imvs)
	
	    return loss, trn_inputs
	
	```
### *SNPEC*.**epochs\_round**`#!py3 (self, epochs)` { #epochs\_round data-toc-label=epochs\_round }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def epochs_round(self, epochs):
	    # number of training examples for this round
	    if type(epochs) == list:
	        try:
	            epochs_round = epochs[self.round-1]
	        except:
	            epochs_round = epochs[-1]
	    else:
	        epochs_round = epochs
	
	    return epochs_round
	
	```
### *SNPEC*.**gen**`#!py3 (self, n_train, project_to_gaussian=False, **kwargs)` { #gen data-toc-label=gen }


```
Generate from generator and z-transform

Parameters
----------
n_samples : int
    Number of samples to generate
n_reps : int
    Number of repeats per parameter
verbose : None or bool or str
    If None is passed, will default to self.verbose
project_to_gaussian: bool
    Whether to always return Gaussian objects (instead of MoG)
n_train: int
    Number of training samples
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen(self, n_train, project_to_gaussian=False, **kwargs):
	    
	    if 'verbose' in kwargs.keys():
	        verbose = kwargs['verbose']
	    else:
	        verbose = self.verbose
	    verbose = '(round {}) '.format(self.round) if verbose else False
	    n_train_round = self.n_train_round(n_train)
	
	    trn_data = super().gen(n_train_round, verbose=verbose, **kwargs)
	    n_train_round = trn_data[0].shape[0]  # may have decreased (rejection)
	
	    return trn_data, n_train_round
	
	```
### *SNPEC*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *SNPEC*.**loss**`#!py3 (self)` { #loss data-toc-label=loss }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	@abc.abstractmethod
	def loss(self):
	    pass
	
	```
### *SNPEC*.**monitor\_dict\_from\_names**`#!py3 (self, monitor=None)` { #monitor\_dict\_from\_names data-toc-label=monitor\_dict\_from\_names }


```
Generate monitor dict from list of variable names
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def monitor_dict_from_names(self, monitor=None):
	    
	    if monitor is not None:
	        observe = {}
	        if isinstance(monitor, str):
	            monitor = [monitor]
	        for m in monitor:
	            if m in self.observables:
	                observe[m] = self.observables[m]
	    else:
	        observe = None
	    return observe
	
	```
### *SNPEC*.**n\_train\_round**`#!py3 (self, n_train)` { #n\_train\_round data-toc-label=n\_train\_round }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def n_train_round(self, n_train):
	    # number of training examples for this round
	    if type(n_train) == list:
	        try:
	            n_train_round = n_train[self.round-1]
	        except:
	            n_train_round = n_train[-1]
	    else:
	        n_train_round = n_train
	
	    return n_train_round
	
	```
### *SNPEC*.**norm\_init**`#!py3 (self)` { #norm\_init data-toc-label=norm\_init }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def norm_init(self):
	    if self.init_norm and self.network.density == 'mog':
	        print('standardizing network initialization')
	        if self.network.n_components > 1:
	            self.standardize_init(fcv = self.init_fcv)
	        else:
	            self.standardize_init(fcv = 0.)
	
	```
### *SNPEC*.**pilot\_run**`#!py3 (self, pilot_samples, n_stats, min_std=0.0001)` { #pilot\_run data-toc-label=pilot\_run }


```
Pilot run in order to find parameters for z-scoring stats
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def pilot_run(self, pilot_samples, n_stats, min_std=1e-4):
	    
	    if pilot_samples is None or \
	            (isint(pilot_samples) and pilot_samples == 0):
	        self.unused_pilot_samples = ([], [])
	        self.stats_mean = np.zeros(n_stats)
	        self.stats_std = np.ones(n_stats)
	        return
	
	    if isint(pilot_samples):  # determine via pilot run
	        assert pilot_samples > 0
	        if self.seed is not None:  # reseed generator for consistent inits
	            self.generator.reseed(self.gen_newseed())
	        verbose = '(pilot run) ' if self.verbose else False
	        params, stats = self.generator.gen(pilot_samples, verbose=verbose)
	    else:  # samples were provided as an input
	        params, stats = pilot_samples
	
	    self.stats_mean = np.nanmean(stats, axis=0)
	    self.stats_std = np.nanstd(stats, axis=0)
	    assert not np.isnan(self.stats_mean).any(), "pilot run failed"
	    assert not np.isnan(self.stats_std).any(), "pilot run failed"
	    self.stats_std[self.stats_std == 0.0] = 1.0
	    self.stats_std = np.maximum(self.stats_std, min_std)
	    assert (self.stats_std > 0).all(), "pilot run failed"
	    ok_sims = np.logical_not(np.logical_or(np.isnan(stats).any(axis=1),
	                                           np.isnan(params).any(axis=1)))
	    self.unused_pilot_samples = (params[ok_sims, :], stats[ok_sims, :])
	
	```
### *SNPEC*.**predict**`#!py3 (self, *args, **kwargs)` { #predict data-toc-label=predict }


```
Predict posterior given x

Parameters
----------
x : array
    Stats for which to compute the posterior
deterministic : bool
    if True, mean weights are used for Bayesian network
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def predict(self, *args, **kwargs):
	    p = super().predict(*args, **kwargs)
	
	    if self.round > 0 and self.proposal_used[-1] in ['gaussian', 'mog']:
	        assert self.network.density == 'mog' and isinstance(p, dd.MoG)
	        P_offset = np.eye(p.ndim) * self.Ptol
	        # add the prior precision to each posterior component if needed
	        if self.add_prior_precision and isinstance(self.generator.prior, dd.Gaussian):
	            P_offset += self.generator.prior.P
	        p = dd.MoG(a=p.a, xs=[dd.Gaussian(m=x.m, P=x.P + P_offset, seed=x.seed) for x in p.xs])
	
	    return p
	
	```
### *SNPEC*.**reinit\_network**`#!py3 (self)` { #reinit\_network data-toc-label=reinit\_network }


```
Reinitializes the network instance (re-setting the weights!)
        
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reinit_network(self):
	    
	    self.network = NeuralNet(**self.kwargs)
	    self.svi = self.network.svi if 'svi' in dir(self.network) else False
	    update self.kwargs['seed'] so that reinitializing the network gives a
	    different result each time unless we reseed the inference method
	    self.kwargs['seed'] = self.gen_newseed()
	    self.norm_init()
	
	```
### *SNPEC*.**remove\_hidden\_biases**`#!py3 (self)` { #remove\_hidden\_biases data-toc-label=remove\_hidden\_biases }


```
Resets all bias weights in hidden layers to zero.

        
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def remove_hidden_biases(self):
	    
	    def idx_hiddens(x):
	        return x.name[0] == 'h'
	
	    for b in filter(idx_hiddens, self.network.mps_bp):
	        b.set_value(np.zeros_like(b.get_value()))
	
	```
### *SNPEC*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


```
reseed inference method's RNG, then generator, then network
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reseed(self, seed):
	    
	    self.rng.seed(seed=seed)
	    self.seed = seed
	    self.kwargs['seed'] = self.gen_newseed()   # for consistent NN init
	    self.generator.reseed(self.gen_newseed())  # also reseeds prior + model
	    if isinstance(self.network, NeuralNet):
	        self.network.reseed(self.gen_newseed())  # for reproducible samples
	
	```
### *SNPEC*.**reset**`#!py3 (self, seed=None)` { #reset data-toc-label=reset }


```
Resets inference method to a naive state, before it has seen any
real or simulated data. The following happens, in order:
1) The generator's proposal is set to None, and self.round is set to 0
2) The inference method is reseeded if a seed is provided
3) The network is reinitialized
4) Any additional resetting of state specific to each inference method
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def reset(self, seed=None):
	    super().reset(seed=seed)
	    self.trn_datasets, self.proposal_used = [], []
	
	```
### *SNPEC*.**run**`#!py3 (self, n_rounds=1, proposal='gaussian', silent_fail=True, **kwargs)` { #run data-toc-label=run }


```
Run algorithm

Parameters
----------
n_train : int or list of ints
    Number of data points drawn per round. If a list is passed, the
    nth list element specifies the number of training examples in the
    nth round. If there are fewer list elements than rounds, the last
    list element is used.
n_rounds : int
    Number of rounds
proposal : str
    Specifier for type of proposal used: continuous ('gaussian', 'mog')
    or 'atomic' proposals are implemented.
epochs : int
    Number of epochs used for neural network training
minibatch : int
    Size of the minibatches used for neural network training
monitor : list of str
    Names of variables to record during training along with the value
    of the loss function. The observables attribute contains all
    possible variables that can be monitored
round_cl : int
    Round after which to start continual learning
stop_on_nan : bool
    If True, will halt if NaNs in the loss are encountered
silent_fail : bool
    If true, will continue without throwing an error when a round fails
kwargs : additional keyword arguments
    Additional arguments for the Trainer instance

Returns
-------
logs : list of dicts
    Dictionaries contain information logged while training the networks
trn_datasets : list of (params, stats)
    training datasets, z-transformed
posteriors : list of distributions
    posterior after each round
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def run(self, n_rounds=1, proposal='gaussian', silent_fail=True, **kwargs):
	    
	    # support 'discrete' instead of 'atomic' for backwards compatibility
	    if proposal == 'discrete':
	        proposal = 'atomic'
	    elif proposal == 'discrete_comb':
	        proposal = 'atomic_comb'
	
	    logs = []
	    trn_datasets = []
	    posteriors = []
	
	    if 'train_on_all' in kwargs.keys() and kwargs['train_on_all'] is True:
	        kwargs['round_cl'] = np.inf
	        if proposal == 'gaussian' and self.network.n_components > 1 and \
	                'reuse_prior_samples' not in kwargs.keys():
	            # prevent numerical instability (broad unused comps)
	            kwargs['reuse_prior_samples'] = False
	
	    for r in range(n_rounds):
	        self.round += 1
	
	        if silent_fail:
	            try:
	                log, trn_data = self.run_round(proposal, **kwargs)
	            except:
	                print('Round {0} failed'.format(self.round))
	                import sys
	                self.exception_info = sys.exc_info()
	                break
	        else:
	            log, trn_data = self.run_round(proposal, **kwargs)
	
	        logs.append(log)
	        trn_datasets.append(trn_data)
	        posteriors.append(self.predict(self.obs))
	
	    return logs, trn_datasets, posteriors
	
	```
### *SNPEC*.**run\_MoG**`#!py3 (self, n_train=100, epochs=100, minibatch=50, n_atoms=None, moo=None, train_on_all=False, round_cl=1, stop_on_nan=False, monitor=None, verbose=False, print_each_epoch=False, reuse_prior_samples=True, patience=20, monitor_every=None, **kwargs)` { #run\_MoG data-toc-label=run\_MoG }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def run_MoG(self, n_train=100, epochs=100, minibatch=50, n_atoms=None, moo=None, train_on_all=False, round_cl=1,
	            stop_on_nan=False, monitor=None, verbose=False, print_each_epoch=False, reuse_prior_samples=True,
	            patience=20, monitor_every=None, **kwargs):
	
	    # simulate data
	    self.set_proposal(project_to_gaussian=False)
	    assert isinstance(self.generator.proposal, dd.MoG)
	    prop = self.generator.proposal.ztrans(self.params_mean, self.params_std)
	
	    trn_data, n_train_round = self.gen(n_train)
	    trn_data = (*trn_data, *MoG_prop_APT_training_vars(prop, n_train_round, prop.n_components))
	
	    self.trn_datasets.append(trn_data)
	
	    if train_on_all:
	        prev_datasets = []
	        for i, d in enumerate(self.trn_datasets):
	            if self.proposal_used[i] == 'mog':
	                prev_datasets.append(d)
	            elif self.proposal_used == 'prior' and reuse_prior_samples:
	                prior = self.generator.prior
	                if not isinstance(prior, dd.Uniform):
	                    prior = prior.ztrans(self.params_mean, self.params_std)
	                d = (*d, *MoG_prop_APT_training_vars(prior, n_train_round))
	                prev_datasets.append(d)
	            elif self.proposal_used[i] == 'gaussian':
	                params, stats, prop_m, prop_P = d
	                if np.diff(prop_m, axis=0).any() or np.diff(prop_P, axis=0).any():
	                    continue  # reusing samples with proposals that changed within a round is not yet supported
	                prop = dd.Gaussian(m=prop_m[0], P=prop_P[0])
	                d = (params, stats, *MoG_prop_APT_training_vars(prop, n_train_round))
	                prev_datasets.append(d)
	            else:  # can't re-use samples from this proposal
	                continue
	
	        trn_data = combine_trn_datasets(prev_datasets)
	        n_train_round = trn_data[0].shape[0]
	
	    self.loss, trn_inputs = self.define_loss(n=n_train_round, round_cl=round_cl, proposal='mog')
	
	    t = Trainer(self.network,
	                self.loss,
	                trn_data=trn_data, trn_inputs=trn_inputs,
	                seed=self.gen_newseed(),
	                monitor=self.monitor_dict_from_names(monitor),
	                **kwargs)
	
	    log = t.train(epochs=self.epochs_round(epochs), minibatch=minibatch, verbose=verbose,
	                  print_each_epoch=print_each_epoch, stop_on_nan=stop_on_nan,
	                  patience=patience, monitor_every=monitor_every)
	
	    return log, trn_data
	
	```
### *SNPEC*.**run\_atomic**`#!py3 (self, n_train=100, epochs=100, minibatch=50, n_atoms=10, moo='resample', train_on_all=False, reuse_prior_samples=True, combined_loss=False, round_cl=1, stop_on_nan=False, monitor=None, patience=20, monitor_every=None, verbose=False, print_each_epoch=False, **kwargs)` { #run\_atomic data-toc-label=run\_atomic }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def run_atomic(self, n_train=100, epochs=100, minibatch=50, n_atoms=10, moo='resample', train_on_all=False,
	               reuse_prior_samples=True, combined_loss=False, round_cl=1, stop_on_nan=False, monitor=None,
	               patience=20, monitor_every=None,
	               verbose=False, print_each_epoch=False, **kwargs):
	
	    # activetrainer doesn't de-norm params before evaluating the prior
	    assert np.all(self.params_mean == 0.0) and np.all(self.params_std == 1.0), "prior_norm + atomic not supported"
	
	    assert minibatch > 1, "minimum minibatch size 2 for atomic proposals"
	    if n_atoms is None:
	        n_atoms = minibatch - 1 if theano.config.device.startswith('cuda') else np.minimum(minibatch - 1, 9)
	    assert n_atoms < minibatch, "Minibatch too small for this many atoms"
	    # simulate data
	    self.set_proposal()
	    trn_data, n_train_round = self.gen(n_train)
	    self.trn_datasets.append(trn_data)  # don't store prior_masks
	
	    if train_on_all:
	        if reuse_prior_samples:
	            trn_data = combine_trn_datasets(self.trn_datasets, max_inputs=2)
	        else:
	            trn_data = combine_trn_datasets(
	                [td for td, pu in zip(self.trn_datasets, self.proposal_used) if pu != 'prior'])
	        if combined_loss:
	            prior_masks = \
	                [np.ones(td[0].shape[0], dtype) * (pu == 'prior')
	                 for td, pu in zip(self.trn_datasets, self.proposal_used)]
	            trn_data = (*trn_data, np.concatenate(prior_masks))
	        n_train_round = trn_data[0].shape[0]
	
	    # train network
	    self.loss, trn_inputs = self.define_loss(n=n_train_round,
	                                             round_cl=round_cl,
	                                             proposal='atomic',
	                                             combined_loss=combined_loss and train_on_all)
	
	    t = ActiveTrainer(self.network,
	                      self.loss,
	                      trn_data=trn_data, trn_inputs=trn_inputs,
	                      seed=self.gen_newseed(),
	                      monitor=self.monitor_dict_from_names(monitor),
	                      generator=self.generator,
	                      n_atoms=n_atoms,
	                      moo=moo,
	                      obs=(self.obs - self.stats_mean) / self.stats_std,
	                      **kwargs)
	
	    log = t.train(epochs=self.epochs_round(epochs), minibatch=minibatch, verbose=verbose,
	                  print_each_epoch=print_each_epoch, strict_batch_size=True,
	                  patience=patience, monitor_every=monitor_every)
	
	    return log, trn_data
	
	```
### *SNPEC*.**run\_gaussian**`#!py3 (self, n_train=100, epochs=100, minibatch=50, n_atoms=None, moo=None, train_on_all=False, round_cl=1, stop_on_nan=False, monitor=None, verbose=False, print_each_epoch=False, patience=20, monitor_every=None, reuse_prior_samples=True, **kwargs)` { #run\_gaussian data-toc-label=run\_gaussian }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def run_gaussian(self, n_train=100, epochs=100, minibatch=50, n_atoms=None, moo=None,  train_on_all=False,
	                 round_cl=1, stop_on_nan=False, monitor=None, verbose=False, print_each_epoch=False,
	                 patience=20, monitor_every=None,
	                 reuse_prior_samples=True, **kwargs):
	
	    # simulate data
	    self.set_proposal(project_to_gaussian=True)
	    assert isinstance(self.generator.proposal, dd.Gaussian)
	    prop = self.generator.proposal.ztrans(self.params_mean, self.params_std)
	
	    trn_data, n_train_round = self.gen(n_train)
	
	    prop_m = np.expand_dims(prop.m, 0).repeat(n_train_round, axis=0)
	    prop_P = np.expand_dims(prop.P, 0).repeat(n_train_round, axis=0)
	    trn_data = (*trn_data, prop_m, prop_P)
	    self.trn_datasets.append(trn_data)
	
	    if train_on_all:
	        prev_datasets = []
	        for i, d in enumerate(self.trn_datasets):
	            if self.proposal_used[i] == 'gaussian':
	                prev_datasets.append(d)
	                continue
	            elif self.proposal_used[i] != 'prior' or not reuse_prior_samples:
	                continue
	            # prior samples. the Gauss loss will reduce to the prior loss
	            if isinstance(self.generator.prior, dd.Gaussian):
	                prior = self.generator.prior.ztrans(self.params_mean, self.params_std)
	                prop_m = prior.mean
	                prop_P = prior.P
	            elif isinstance(self.generator.prior, dd.Uniform):
	                # model a uniform as an zero-precision Gaussian:
	                prop_m = np.zeros(self.generator.prior.ndim, dtype)
	                prop_P = np.zeros((self.generator.prior.ndim, self.generator.prior.ndim), dtype)
	            else:  # can't reuse prior samples unless prior is uniform or Gaussian
	                continue
	            prop_m = np.expand_dims(prop_m, 0).repeat(d[0].shape[0], axis=0)
	            prop_P = np.expand_dims(prop_P, 0).repeat(d[0].shape[0], axis=0)
	            prev_datasets.append((*d, prop_m, prop_P))
	
	        trn_data = combine_trn_datasets(prev_datasets)
	        n_train_round = trn_data[0].shape[0]
	
	    # train network
	    self.loss, trn_inputs = self.define_loss(n=n_train_round,
	                                             round_cl=round_cl,
	                                             proposal='gaussian')
	    t = Trainer(self.network,
	                self.loss,
	                trn_data=trn_data, trn_inputs=trn_inputs,
	                seed=self.gen_newseed(),
	                monitor=self.monitor_dict_from_names(monitor),
	                **kwargs)
	
	    log = t.train(epochs=self.epochs_round(epochs), minibatch=minibatch, verbose=verbose,
	                  print_each_epoch=print_each_epoch, stop_on_nan=stop_on_nan,
	                  patience=patience, monitor_every=monitor_every)
	
	    return log, trn_data
	
	```
### *SNPEC*.**run\_prior**`#!py3 (self, n_train=100, epochs=100, minibatch=50, n_atoms=None, moo=None, train_on_all=False, round_cl=1, stop_on_nan=False, monitor=None, verbose=False, print_each_epoch=False, patience=20, monitor_every=None, reuse_prior_samples=True, **kwargs)` { #run\_prior data-toc-label=run\_prior }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def run_prior(self, n_train=100, epochs=100, minibatch=50, n_atoms=None,
	              moo=None, train_on_all=False, round_cl=1, stop_on_nan=False,
	              monitor=None, verbose=False, print_each_epoch=False,
	              patience=20, monitor_every=None, reuse_prior_samples=True,
	              **kwargs):
	
	    # simulate data
	    self.generator.proposal = self.generator.prior
	    trn_data, n_train_round = self.gen(n_train)
	    self.trn_datasets.append(trn_data)
	
	    if train_on_all and reuse_prior_samples:
	        prior_datasets = [d for i, d in enumerate(self.trn_datasets)
	                          if self.proposal_used[i] == 'prior']
	        trn_data = combine_trn_datasets(prior_datasets)
	        n_train_round = trn_data[0].shape[0]
	
	    # train network
	    self.loss, trn_inputs = self.define_loss(n=n_train_round,
	                                             round_cl=round_cl,
	                                             proposal='prior')
	    t = Trainer(self.network,
	                self.loss,
	                trn_data=trn_data, trn_inputs=trn_inputs,
	                seed=self.gen_newseed(),
	                monitor=self.monitor_dict_from_names(monitor),
	                **kwargs)
	    log = t.train(epochs=self.epochs_round(epochs), minibatch=minibatch,
	                  verbose=verbose, print_each_epoch=print_each_epoch,
	                  stop_on_nan=stop_on_nan, patience=patience, monitor_every=monitor_every)
	
	    return log, trn_data
	
	```
### *SNPEC*.**run\_repeated**`#!py3 (self, n_repeats=10, n_NN_inits_per_repeat=1, callback=None, **kwargs)` { #run\_repeated data-toc-label=run\_repeated }


```
Repeatedly run the method and collect results. Optionally, carry out
several runs with the same initial generator RNG state but different
neural network initializations.

parameters
----------
n_repeats : int
    Number of times to run the algorithm
n_NN_inits : int
    Number of times to
callback: function
    callback function that will be called after each run. It should
    take 4 inputs: callback(log, train_data, posterior, self)
kwargs : additional keyword arguments
    Additional arguments that will be passed to the run() method
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def run_repeated(self, n_repeats=10, n_NN_inits_per_repeat=1,
	                 callback=None, **kwargs):
	    
	    posteriors, outputs, repeat_index = [], [], []
	    for r in range(n_repeats):
	
	        if n_NN_inits_per_repeat > 1:
	            generator_seed = self.gen_newseed()
	
	        for i in range(n_NN_inits_per_repeat):
	
	            self.reset()
	            if n_NN_inits_per_repeat > 1:
	                self.generator.reseed(generator_seed)
	
	            log, train_data, posterior = self.run(**kwargs)
	
	            if callback is not None:
	                outputs.append(callback(log, train_data, posterior, self))
	            else:
	                outputs.append(None)
	            posteriors.append(posterior)
	            repeat_index.append(r)
	
	    return posteriors, outputs, repeat_index
	
	```
### *SNPEC*.**run\_round**`#!py3 (self, proposal=None, **kwargs)` { #run\_round data-toc-label=run\_round }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def run_round(self, proposal=None, **kwargs):
	
	    proposal = proposal.lower() if self.round > 1 else 'prior'
	    self.proposal_used.append(proposal)
	
	    if proposal == 'prior' or self.round == 1:
	        return self.run_prior(**kwargs)
	    elif proposal == 'gaussian':
	        return self.run_gaussian(**kwargs)
	    elif proposal == 'mog':
	        return self.run_MoG(**kwargs)
	    elif proposal == 'atomic':
	        return self.run_atomic(combined_loss=False, **kwargs)
	    elif proposal == 'atomic_comb':
	        return self.run_atomic(combined_loss=True, **kwargs)
	    else:
	        raise NotImplemented()
	
	```
### *SNPEC*.**set\_proposal**`#!py3 (self, project_to_gaussian=False)` { #set\_proposal data-toc-label=set\_proposal }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def set_proposal(self, project_to_gaussian=False):
	    # posterior estimate becomes new proposal prior
	    if self.round == 0:
	        return None
	
	    posterior = self.predict(self.obs)
	
	    if project_to_gaussian:
	        assert self.network.density == 'mog', "cannot project a MAF"
	        posterior = posterior.project_to_gaussian()
	
	    self.generator.proposal = posterior
	
	```
### *SNPEC*.**standardize\_init**`#!py3 (self, fcv=0.8)` { #standardize\_init data-toc-label=standardize\_init }


```
Standardizes the network initialization on obs

Ensures output distributions for xo have mean zero and unit variance.
Alters hidden layers to propagates x=xo as zero to the last layer, and
alters the MoG layers to produce the desired output distribution.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def standardize_init(self, fcv = 0.8):
	    
	    assert isinstance(self.network, NeuralNet)
	
	    # ensure x' = x - xo
	    self.centre_on_obs()
	
	    # ensure x' = 0 stays zero up to MoG layer (setting biases to zero)
	    self.remove_hidden_biases()
	
	    # ensure MoG returns standardized output on x' = 0
	    self.conditional_norm(fcv)
	
	```
