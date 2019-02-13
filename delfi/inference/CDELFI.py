import delfi.distribution as dd
import numpy as np
import theano.tensor as tt

from delfi.inference.BaseInference import BaseInference
from delfi.distribution.mixture.BaseMixture import BaseMixture
from delfi.distribution import Gaussian, Uniform
from delfi.neuralnet.NeuralNet import NeuralNet
from delfi.neuralnet.Trainer import Trainer
from delfi.neuralnet.loss.regularizer import svi_kl_zero

from delfi.utils.sbc import SBC

import lasagne.layers as ll
import theano
dtype = theano.config.floatX

def per_round(y):

    if type(y) == list:
        try:
            y_round = y[r-1]
        except:
            y_round = y[-1]
    else:
        y_round = y

    return y_round

def logdet(M):
    slogdet = np.linalg.slogdet(M)
    return slogdet[0] * slogdet[1]

class CDELFI(BaseInference):
    def __init__(self, generator, obs, reg_lambda=0.01, 
                 **kwargs):
        """Conditional density estimation likelihood-free inference (CDE-LFI)

        Implementation of algorithms 1 and 2 of Papamakarios and Murray, 2016.

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
        reg_lambda : float
            Precision parameter for weight regularizer if svi is True
        seed : int or None
            If provided, random number generator will be seeded
        verbose : bool
            Controls whether or not progressbars are shown
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
        """
        # Algorithm 1 of PM requires a single component
        #if 'n_components' in kwargs:
        #    assert kwargs['n_components'] == 1 # moved n_components argument to run()
        self.obs = obs
        
        super().__init__(generator, **kwargs)

        self.init_fcv = 0.8 # CDELFI won't call conditional_norm() with single component

        if np.any(np.isnan(self.obs)):
            raise ValueError("Observed data contains NaNs")

        self.reg_lambda = reg_lambda
        self.round = 0 # total round counter

    def loss(self, N):
        """Loss function for training

        Parameters
        ----------
        N : int
            Number of training samples
        """
        loss = -tt.mean(self.network.lprobs)

        if self.svi:

            if self.reg_lambda > 0:
                kl, imvs = svi_kl_zero(self.network.mps, self.network.sps,
                                       self.reg_lambda)
            else:
                kl, imvs = 0, {}

            loss = loss + 1 / N * kl

            # adding nodes to dict s.t. they can be monitored
            self.observables['loss.kl'] = kl
            self.observables.update(imvs)

        return loss

    def run(self, n_train=100, n_rounds=2, epochs=100, minibatch=50,
            monitor=None, n_components=1, stndrd_comps=False, 
            project_proposal=False, sbc_fun = None,
            **kwargs):
        """Run algorithm

        Parameters
        ----------
        n_train : int or list of ints
            Number of data points drawn per round. If a list is passed, the
            nth list element specifies the number of training examples in the
            nth round. If there are fewer list elements than rounds, the last
            list element is used.
        n_rounds : int
            Number of rounds
        epochs: int
            Number of epochs used for neural network training
        minibatch: int
            Size of the minibatches used for neural network training
        monitor : list of str
            Names of variables to record during training along with the value
            of the loss function. The observables attribute contains all
            possible variables that can be monitored
        n_components : int
            Number of components in final round (if > 1, gives PM's algorithm 2)
        kwargs : additional keyword arguments
            Additional arguments for the Trainer instance

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged while training the networks
        trn_datasets : list of (params, stats)
            training datasets, z-transformed
        posteriors : list of posteriors
            posterior after each round
        """
        logs = []
        trn_datasets = []
        posteriors = []

        #assert self.kwargs['n_components'] == 1 
        # could also allow to go back to single Gaussian via project_to_gaussian()

        for r in range(1, n_rounds + 1):  # start at 1

            self.round += 1

            if self.round > 1:
                # posterior becomes new proposal prior
                proposal = self.predict(self.obs)
                if isinstance(proposal, BaseMixture) and (len(proposal.xs)==1 or project_proposal):  
                    proposal = proposal.project_to_gaussian()
                self.generator.proposal = proposal 

            # number of training examples for this round
            epochs_round = per_round(epochs)
            n_train_round = per_round(n_train)

            # draw training data (z-transformed params and stats)
            verbose = '(round {}) '.format(self.round) if self.verbose else False
            trn_data = self.gen(n_train_round, verbose=verbose)[:2]

            if r == n_rounds: 
                self.kwargs.update({'n_components': n_components})
                self.split_components(standardize=stndrd_comps)

            if r > 1:
                self.reinit_network() # reinits network if flag is set


            if hasattr(self.network, 'extra_stats'):
                trn_inputs = [self.network.params, self.network.stats, self.network.extra_stats]
            else:
                trn_inputs = [self.network.params, self.network.stats]

            t = Trainer(self.network, self.loss(N=n_train_round),
                        trn_data=trn_data, trn_inputs=trn_inputs,
                        monitor=self.monitor_dict_from_names(monitor),
                        seed=self.gen_newseed(), **kwargs)
            logs.append(t.train(epochs=epochs_round, minibatch=minibatch,
                                verbose=verbose,n_inputs=self.network.n_inputs,
                                n_inputs_hidden=self.network.n_inputs_hidden))
            trn_datasets.append(trn_data)

            try:
                posteriors.append(self.predict(self.obs))
            except:
                posteriors.append(None)
                print('analytical correction broke !')
                break


            if not sbc_fun is None:
                print('computing simulation-based calibration')
                sbc = SBC(generator=self.generator, inf=self, f=sbc_fun)
                data = (trn_data[0]*self.params_std+self.params_mean,
                        trn_data[1])
                logs[-1]['sbc'] = sbc.test(N=None, L=100, data=data)

        return logs, trn_datasets, posteriors


    def predict(self, x, threhold=0.01):
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        threshold: float
            Threshold for pruning MoG components (percent of posterior mass)
        """
        proposal = self.generator.proposal

        if proposal is None or isinstance(proposal, (Uniform,Gaussian)):  
            posterior = self._predict_from_Gaussian_prop(self.obs)
        elif len(self.generator.proposal.xs) <= self.network.n_components:                    
            print('correcting for MoG proposal')
            posterior = self._predict_from_MoG_prop(self.obs)
        else:
            raise NotImplementedError

        return posterior

    def _predict_from_Gaussian_prop(self, x, threshold=0.01):
        """Predict posterior given x

        Predicts posteriors from the attached MDN and corrects for
        missmatch in the prior and proposal prior if the latter is
        given by a Gaussian object.

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        threshold: float
            Threshold for pruning MoG components (percent of posterior mass)
        """
        if self.generator.proposal is None:
            # no correction necessary
            return super(CDELFI, self).predict(x)  # via super
        else:
            # mog is posterior given proposal prior
            mog = super(CDELFI, self).predict(x)  # via super
            mog.prune_negligible_components(threshold=threshold)

            # compute posterior given prior by analytical division step
            if 'Uniform' in str(type(self.generator.prior)):
                posterior = mog / self.generator.proposal
            elif 'Gaussian' in str(type(self.generator.prior)):
                try: 
                    posterior = (mog * self.generator.prior) / \
                        self.generator.proposal
                except: 
                    print('analytical correction seemingly broke.')
                    print('attempting to save the prediction.')
                    posterior = self._backup_predict_from_Gaussian_prop(mog)
            else:
                raise NotImplemented

            return posterior

    def _backup_predict_from_Gaussian_prop(self, mog, thresh=0.):
        """Predict posterior given x

        Predicts posteriors from the attached MDN and corrects for
        missmatch in the prior and proposal prior if the latter is
        given by a Gaussian object.

        Can still be used if the result would have improper covariances: 
        Will try to replace directions of negative precisions in the 
        posterior with proposal precisions. 

        Parameters
        ----------
        mog : mixture of Gaussian object
            Uncorrected MoG posterior estimate
        thresh : float
            Threshold for precisions of the MoG components.
            Pick >0 for added numerical stability. 
        """

        proposal, prior = self.generator.proposal, self.generator.prior
        assert isinstance(proposal, Gaussian) and isinstance(prior, Gaussian)

        xs_new = []
        for c in mog.xs:

            # corrected precision matrix
            Pc = c.P - proposal.P + prior.P

            # spectrum and eigenvectors of corrected precision matrix
            Lu, Q = np.linalg.eig(Pc)
            # precisions along eigenvectors of corrected precision matrix
            Lp = np.diag((Q.T.dot(proposal.P).dot(Q)))

            # identify degenerate precisions
            idx = np.where(Lu <= thresh)[0]

            # replace degenerate precisions with those from proposal
            L = Lu.copy()
            if idx.size > 0:
                L[idx] = np.maximum( Lp[idx], thresh )

            # recompute means and covariances
            S = Q.dot(np.diag(1./L)).dot(Q.T)
            m = S.dot(c.Pm - proposal.Pm + prior.Pm) 

            xs_new.append(Gaussian(m=m, S=S))

        return dd.MoG(xs = xs_new, a = mog.a)


    def _predict_from_MoG_prop(self, x, threshold=0.01):
        """Predict posterior given x

        Predicts posteriors from the attached MDN and corrects for
        missmatch in the prior and proposal prior if the latter is
        given by a Gaussian mixture with multiple mixture components.

        Assumes proposal mixture components are well-separated, which 
        allows to locally correct each posterior component only for the 
        closest proposal component. 

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        threshold: float
            Threshold for pruning MoG components (percent of posterior mass)
        """
        # mog is posterior given proposal prior
        mog = super(CDELFI, self).predict(x)  # via super

        proposal, prior = self.generator.proposal, self.generator.prior
        assert isinstance(prior, Gaussian)

        ldetP0, d0  = logdet(prior.P), prior.m.dot(prior.Pm)
        means = np.vstack([c.m for c in proposal.xs])

        xs_new, a_new = [], []
        for c, j in zip(mog.xs, np.arange(mog.a.size)):

            # greedily pairing proposal and posterior components by means
            # (should probably at least use Mahalanobis distance)
            dists = np.sum( (means - np.atleast_2d(c.m))**2, axis=1)
            i = np.argmin(dists)

            c_prop = proposal.xs[i]
            a_prop = proposal.a[i]

            # correct means and covariances of individual proposals
            c_post = (c * prior) / c_prop

            # correct mixture coefficients a[i]

            # prefactors
            log_a = np.log(mog.a[j]) - np.log(a_prop) 
            # determinants
            log_a += 0.5 * (logdet(c.P)+ldetP0-logdet(c_prop.P)-logdet(c_post.P))
            # Mahalanobis distances
            log_a -= 0.5 * c.m.dot(c.Pm)
            log_a -= 0.5 * d0
            log_a += 0.5 * c_prop.m.dot(c_prop.Pm)
            log_a += 0.5 * c_post.m.dot(c_post.Pm)
            a_i = np.exp(log_a)
            
            xs_new.append(c_post)
            a_new.append(a_i)

        a_new = np.array(a_new)
        # alpha defined only up to \tilde{p}(x) / p(x), i.e. need to normalize
        a_new /= a_new.sum()

        mog = dd.MoG( xs = xs_new, a = a_new )
        mog.prune_negligible_components(threshold=threshold)

        return mog


    def predict_uncorrected(self, x):
        """Predict posterior given x under proposal prior

        Predicts the uncorrected posterior associated with the proposal
        prior (versus the original prior). 

        Allows to obtain some posterior estimates when the analytical 
        correction for the proposal prior fails. 

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        """

        return super(CDELFI, self).predict(x)  # via super

    def split_components(self, standardize=False):
        """Split MoG components

        Replicates, perturbes and (optionally) standardizes MoG
        components across rounds. 

        Replica MoG components serve as part of initialization for 
        MDN in the next round. 

        Parameters
        ----------
        standardize : boolean
            whether to standardize the replicated components 
        """
        # define magnitute of (symmetry-breaking) perturbation noise on weights
        eps = 1.0e-2 if standardize else 1.0e-6

        if self.kwargs['n_components'] > self.network.n_components:
            
            # get parameters of current network
            old_params = self.network.params_dict.copy()
            old_n_components = self.network.n_components

            # create new network
            self.network = NeuralNet(**self.kwargs)
            new_params = self.network.params_dict

            # set weights of new network
            comps_new = [s for s in new_params if 'means' in s or
                      'precisions' in s]

            # weights of additional components are duplicates
            for p in np.sort(comps_new):
                i = int(float(p[-1]))%old_n_components
                # WARNING: this assumes there is less <10 old components!
                old_params[p] = old_params[p[:-1] + str(i)].copy()
                old_params[p] += eps*self.rng.randn(*new_params[p].shape)

            # assert mixture coefficients are alpha_k = 1/n_components)
            old_params['weights.mW'] = 0. * new_params['weights.mW']
            old_params['weights.mb'] = 0. * new_params['weights.mb']

            self.network.params_dict = old_params

            if standardize:
                self.standardize_components()


    def standardize_components(self):
        """Standardize MoG components

        Changes weights in MoG layer of the attached MDN to split the 
        support of a Gaussian proposal prior among multiple MoG components
        of the posterior estimate. 

        Meant to give multi-component MDN initializations that start with
        more distinguishable components than is achieved through simply 
        replicating components and perturbing them with symmetry-breaking
        noise.  

        """
        assert isinstance(self.generator.proposal, Gaussian) 

        # grab activation from last hidden layer
        last_hidden = self.network.layer['mixture_weights'].input_layer
        h = theano.function(
            inputs=[self.network.stats, self.network.extra_stats],
            outputs=[ll.get_output(last_hidden)])

        n, d = self.network.n_inputs_hidden, self.network.n_inputs
        stats = (self.obs-self.stats_mean)/self.stats_std
        if n > 0:
            input_stats = [stats[:,:-n].reshape(-1,*d).astype(dtype),
                           stats[:,-n:].astype(dtype)]
        else: 
            input_stats = stats.reshape(-1, *d).astype(dtype)

        # target variance and (z-scored) mean from proposal
        proposal = self.generator.proposal

        self.conditional_norm(fcv=self.init_fcv,
                              tmu=(proposal.m-self.params_mean)/self.params_std,
                              tSig=proposal.S,
                              h=h(*input_stats)[0].flatten())
