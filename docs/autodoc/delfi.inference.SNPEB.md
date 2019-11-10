## **SNPEB**`#!py3 class` { #SNPEB data-toc-label=SNPEB }


### *SNPEB*.**\_\_init\_\_**`#!py3 (self, generator, obs, prior_norm=False, pilot_samples=100, convert_to_T=3, reg_lambda=0.01, prior_mixin=0, kernel=None, seed=None, verbose=True, **kwargs)` { #\_\_init\_\_ data-toc-label=\_\_init\_\_ }


```
SNPE-B

Implementation of Lueckmann, Goncalves, Bassetto, Öcal, Nonnenmacher & Macke (NeurIPS 2017)

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
convert_to_T : None or int
    Convert proposal distribution to Student's T? If a number if given,
    the number specifies the degrees of freedom. None for no conversion
reg_lambda : float
    Precision parameter for weight regularizer if svi is True
prior_mixin : float
    Percentage of the prior mixed into the proposal prior. While training,
    an additional prior_mixin * N samples will be drawn from the actual prior
    in each round.
seed : int or None
    If provided, random number generator will be seeded
verbose : bool
    Controls whether or not progressbars are shown
kwargs : additional keyword arguments
    Additional arguments for the NeuralNet instance, including:
        n_components : int
            Number of components of the mixture density
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
	def __init__(self, generator, obs, prior_norm=False, pilot_samples=100,
	             convert_to_T=3, reg_lambda=0.01, prior_mixin=0, kernel=None, seed=None, verbose=True,
	             **kwargs):
	    
	    assert obs is not None, "SNPE requires observed data"
	    self.obs = np.asarray(obs)
	    super().__init__(generator, prior_norm=prior_norm,
	                     pilot_samples=pilot_samples, seed=seed,
	                     verbose=verbose, **kwargs)
	
	    if np.any(np.isnan(self.obs)):
	        raise ValueError("Observed data contains NaNs")
	
	    self.reg_lambda = reg_lambda
	    self.convert_to_T = convert_to_T
	
	    self.prior_mixin = 0 if prior_mixin is None else prior_mixin
	
	    self.kernel = kernel
	
	```
### *SNPEB*.**centre\_on\_obs**`#!py3 (self)` { #centre\_on\_obs data-toc-label=centre\_on\_obs }


```
Centres first-layer input onto observed summary statistics

Ensures x' = x - xo, i.e. first-layer input x' = 0 for x = xo.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def centre_on_obs(self):
	    
	
	    self.stats_mean = self.obs.copy()
	
	```
### *SNPEB*.**compile\_observables**`#!py3 (self)` { #compile\_observables data-toc-label=compile\_observables }


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
### *SNPEB*.**conditional\_norm**`#!py3 (self, fcv=0.8, tmu=None, tSig=None, h=None)` { #conditional\_norm data-toc-label=conditional\_norm }


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
### *SNPEB*.**gen**`#!py3 (self, n_samples, n_reps=1, prior_mixin=0, verbose=None)` { #gen data-toc-label=gen }


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
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def gen(self, n_samples, n_reps=1, prior_mixin=0, verbose=None):
	    
	    assert n_reps == 1, 'n_reps > 1 is not yet supported'
	    verbose = self.verbose if verbose is None else verbose
	
	    n_pilot = np.minimum(n_samples, len(self.unused_pilot_samples[0]))
	    if n_pilot > 0 and self.generator.proposal is self.generator.prior:  # reuse pilot samples
	        params = self.unused_pilot_samples[0][:n_pilot, :]
	        stats = self.unused_pilot_samples[1][:n_pilot, :]
	        self.unused_pilot_samples = \
	            (self.unused_pilot_samples[0][n_pilot:, :],
	             self.unused_pilot_samples[1][n_pilot:, :])
	        n_samples -= n_pilot
	
	        if n_samples > 0:
	            params_rem, stats_rem = self.generator.gen(n_samples,
	                                                       prior_mixin=prior_mixin,
	                                                       verbose=verbose)
	            params = np.concatenate((params, params_rem), axis=0)
	            stats = np.concatenate((stats, stats_rem), axis=0)
	    else:
	        params, stats = self.generator.gen(n_samples,
	                                           prior_mixin=prior_mixin,
	                                           verbose=verbose)
	
	    # z-transform params and stats
	    params = (params - self.params_mean) / self.params_std
	    stats = (stats - self.stats_mean) / self.stats_std
	
	    return params, stats
	
	```
### *SNPEB*.**gen\_newseed**`#!py3 (self)` { #gen\_newseed data-toc-label=gen\_newseed }


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
### *SNPEB*.**loss**`#!py3 (self, N, round_cl=1)` { #loss data-toc-label=loss }


```
Loss function for training

Parameters
----------
N : int
    Number of training samples
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def loss(self, N, round_cl=1):
	    
	    loss = self.network.get_loss()
	
	    # adding nodes to dict s.t. they can be monitored during training
	    self.observables['loss.lprobs'] = self.network.lprobs
	    self.observables['loss.iws'] = self.network.iws
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
	
	        loss = loss + 1 / N * kl
	
	        # adding nodes to dict s.t. they can be monitored
	        self.observables['loss.kl'] = kl
	        self.observables.update(imvs)
	
	    return loss
	
	```
### *SNPEB*.**monitor\_dict\_from\_names**`#!py3 (self, monitor=None)` { #monitor\_dict\_from\_names data-toc-label=monitor\_dict\_from\_names }


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
### *SNPEB*.**norm\_init**`#!py3 (self)` { #norm\_init data-toc-label=norm\_init }



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
### *SNPEB*.**pilot\_run**`#!py3 (self, pilot_samples, n_stats, min_std=0.0001)` { #pilot\_run data-toc-label=pilot\_run }


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
### *SNPEB*.**predict**`#!py3 (self, x, deterministic=True)` { #predict data-toc-label=predict }


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
	def predict(self, x, deterministic=True):
	    
	    assert isinstance(self.network, NeuralNet)
	    # z-transform inputs
	    x_zt = (x - self.stats_mean) / self.stats_std
	
	    posterior = self.network.get_density(x_zt, deterministic=deterministic)
	
	    # z-transform outputs
	    if self.network.density == 'mog':
	        posterior = posterior.ztrans_inv(self.params_mean, self.params_std)
	    elif self.network.density == 'maf':
	        posterior.set_scale_and_offset(scale=self.params_std,
	                                       offset=self.params_mean)
	    else:
	        assert np.all(self.params_std == 1.0) and \
	               np.all(self.params_mean == 0.0)
	
	    return posterior
	
	```
### *SNPEB*.**reinit\_network**`#!py3 (self)` { #reinit\_network data-toc-label=reinit\_network }


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
### *SNPEB*.**remove\_hidden\_biases**`#!py3 (self)` { #remove\_hidden\_biases data-toc-label=remove\_hidden\_biases }


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
### *SNPEB*.**reseed**`#!py3 (self, seed)` { #reseed data-toc-label=reseed }


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
### *SNPEB*.**reset**`#!py3 (self, seed=None)` { #reset data-toc-label=reset }


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
	    
	    self.generator.proposal = None
	    self.round = 0
	    if seed is not None:
	        self.reseed(seed)
	    self.reinit_network()
	
	```
### *SNPEB*.**run**`#!py3 (self, n_train=100, n_rounds=2, epochs=100, minibatch=50, round_cl=1, stop_on_nan=False, proposal=None, monitor=None, **kwargs)` { #run data-toc-label=run }


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
proposal : Distribution of None
    If given, will use this distribution as the starting proposal prior
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
	def run(self, n_train=100, n_rounds=2, epochs=100, minibatch=50,
	        round_cl=1, stop_on_nan=False, proposal=None,
	        monitor=None, **kwargs):
	    
	    logs = []
	    trn_datasets = []
	    posteriors = []
	
	    for r in range(n_rounds):
	        self.round += 1
	
	        if r == 0 and proposal is not None:
	            self.generator.proposal = proposal
	        # if round > 1, set new proposal distribution before sampling
	        elif self.round > 1:
	            # posterior becomes new proposal prior
	            proposal = self.predict(self.obs)  # see super
	
	            # convert proposal to student's T?
	            if self.convert_to_T is not None:
	                if type(self.convert_to_T) == int:
	                    dofs = self.convert_to_T
	                else:
	                    dofs = 10
	                proposal = proposal.convert_to_T(dofs=dofs)
	
	            self.generator.proposal = proposal
	
	        # number of training examples for this round
	        if type(n_train) == list:
	            try:
	                n_train_round = n_train[self.round-1]
	            except:
	                n_train_round = n_train[-1]
	        else:
	            n_train_round = n_train
	
	
	        # draw training data (z-transformed params and stats)
	        verbose = '(round {}) '.format(self.round) if self.verbose else False
	
	        trn_data = self.gen(n_train_round, prior_mixin=self.prior_mixin, verbose=verbose)
	        n_train_round = trn_data[0].shape[0]
	
	        # precompute importance weights
	        if self.generator.proposal is not None:
	            params = self.params_std * trn_data[0] + self.params_mean
	            p_prior = self.generator.prior.eval(params, log=False)
	            p_proposal = self.generator.proposal.eval(params, log=False)
	            iws = p_prior / (self.prior_mixin * p_prior + (1 - self.prior_mixin) * p_proposal)
	        else:
	            iws = np.ones((n_train_round,))
	
	        # normalize weights
	        iws /= np.mean(iws)
	
	        if self.kernel is not None:
	            iws *= self.kernel.eval(trn_data[1].reshape(n_train_round, -1))
	
	        trn_data = (trn_data[0], trn_data[1], iws)
	        trn_inputs = [self.network.params, self.network.stats,
	                      self.network.iws]
	
	        t = Trainer(self.network,
	                    self.loss(N=n_train_round, round_cl=round_cl),
	                    trn_data=trn_data, trn_inputs=trn_inputs,
	                    seed=self.gen_newseed(),
	                    monitor=self.monitor_dict_from_names(monitor),
	                    **kwargs)
	        logs.append(t.train(epochs=epochs, minibatch=minibatch,
	                            verbose=verbose, stop_on_nan=stop_on_nan))
	
	        trn_datasets.append(trn_data)
	
	        try:
	            posteriors.append(self.predict(self.obs))
	        except np.linalg.LinAlgError:
	            posteriors.append(None)
	            print("Cannot predict posterior after round {} due to NaNs".format(r))
	            break
	
	    return logs, trn_datasets, posteriors
	
	```
### *SNPEB*.**run\_repeated**`#!py3 (self, n_repeats=10, n_NN_inits_per_repeat=1, callback=None, **kwargs)` { #run\_repeated data-toc-label=run\_repeated }


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
### *SNPEB*.**standardize\_init**`#!py3 (self, fcv=0.8)` { #standardize\_init data-toc-label=standardize\_init }


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
