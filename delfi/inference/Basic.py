import theano.tensor as tt

from delfi.inference.BaseInference import BaseInference
from delfi.neuralnet.Trainer import Trainer
from delfi.neuralnet.loss.regularizer import svi_kl_zero


class Basic(BaseInference):
    def __init__(self, generator, obs=None, prior_norm=False, pilot_samples=100,
                 reg_lambda=0.01, seed=None, verbose=True, **kwargs):
        """Basic inference algorithm

        Uses samples from the prior for density estimation LFI. Network can be
        trained with SVI (optional).

        Parameters
        ----------
        generator : generator instance
            Generator instance
        obs : array or None
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
        """
        super().__init__(generator, prior_norm=prior_norm,
                         pilot_samples=pilot_samples, seed=seed,
                         verbose=verbose, **kwargs)
        self.obs = obs
        self.reg_lambda = reg_lambda

    def loss(self, N):
        """Loss function for training

        Parameters
        ----------
        N : int
            Number of training samples
        """
        loss = -tt.mean(self.network.lprobs)

        if self.svi:
            # keep weights close to zero-centered prior
            kl, _ = svi_kl_zero(self.network.mps, self.network.sps,
                                self.reg_lambda)
            loss = loss + 1/N * kl

        return loss

    def run(self, n_train=100, epochs=100, minibatch=50, monitor=None,
            **kwargs):
        """Run algorithm

        Generate training data using the generator. Set up the Trainer with a
        neural net, a loss function and the generated training data. Train the
        network with the specified training arguments.

        Parameters
        ----------
        n_train : int
            Number of training samples
        epochs : int
            Number of epochs used for neural network training
        minibatch : int
            Size of the minibatches used for neural network training
        monitor : list of str
            Names of variables to record during training along with the value
            of the loss function. The observables attribute contains all
            possible variables that can be monitored
        kwargs : additional keyword arguments
            Additional arguments for the Trainer instance

        Returns
        -------
        log: dict
            dict containing the loss values as returned by Trainer.train()
        trn_data : (params, stats)
            training dataset, z-transformed
        posterior : distribution or None
            posterior for obs if obs is not None
        """
        trn_data = self.gen(n_train, verbose=self.verbose)  # z-transformed
        trn_inputs = [self.network.params, self.network.stats]

        t = Trainer(self.network, self.loss(N=n_train),
                    trn_data=trn_data, trn_inputs=trn_inputs,
                    monitor=self.monitor_dict_from_names(monitor),
                    seed=self.gen_newseed(), **kwargs)
        log = t.train(epochs=epochs, minibatch=minibatch, verbose=self.verbose)

        posterior = self.predict(self.obs) if self.obs is not None else None

        return log, trn_data, posterior
