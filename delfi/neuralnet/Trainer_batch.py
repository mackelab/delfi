import lasagne.updates as lu
import numpy as np
import theano
import theano.tensor as tt

from delfi.utils.progress import no_tqdm, progressbar

dtype = theano.config.floatX


class Trainer_rebatch:
    def __init__(self, network, loss, trn_data, trn_inputs,
                 step=lu.adam, lr=0.001, lr_decay=1.0, max_norm=0.1,
                 monitor=None, seed=None):
        """Construct and configure the trainer

        The trainer takes as inputs a neural network, a loss function and
        training data. During init the theano functions for training are
        compiled.

        Parameters
        ----------
        network : NeuralNet instance
            The neural network to train
        loss : theano variable
            Loss function to be computed for network training
        trn_data : tuple of arrays
            Training data in the form (params, stats)
        trn_inputs : list of theano variables
            Theano variables that should contain the the training data
        step : function
            Function to call for updates, will pass gradients and parameters
        lr : float
            initial learning rate
        lr_decay : float
            learning rate decay factor, learning rate for each epoch is
            set to lr * (lr_decay**epoch)
        max_norm : float
            Total norm constraint for gradients
        monitor : dict
            Dict containing theano variables (and names as keys) that should be
            recorded during training along with the loss function
        seed : int or None
            If provided, random number generator for batches will be seeded
        """
        self.network = network
        self.loss = loss
        self.trn_data = trn_data
        self.trn_inputs = trn_inputs

        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

        # gradients
        grads = tt.grad(self.loss, self.network.aps)
        if max_norm is not None:
            grads = lu.total_norm_constraint(grads, max_norm=max_norm)

        # updates
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_op = theano.shared(np.array(self.lr, dtype=dtype))
        self.updates = step(grads, self.network.aps, learning_rate=self.lr_op)

        # check trn_data
        n_trn_data_list = set([x.shape[0] for x in trn_data])
        assert len(n_trn_data_list) == 1, 'trn_data elements got different len'
        self.n_trn_data = trn_data[0].shape[0]

        # outputs
        self.trn_outputs_names = ['loss']
        self.trn_outputs_nodes = [self.loss]
        if monitor is not None and len(monitor) > 0:
            monitor_names, monitor_nodes = zip(*monitor.items())
            self.trn_outputs_names += monitor_names
            self.trn_outputs_nodes += monitor_nodes

        # function for single update
        self.make_update = theano.function(
            inputs=self.trn_inputs,
            outputs=self.trn_outputs_nodes,
            updates=self.updates
        )

        # initialize variables
        self.loss = float('inf')

    def train(self,
              epochs=250,
              minibatch=50,
              monitor_every=None,
              stop_on_nan=False,
              tol=None,
              verbose=False):
        """Trains the model

        Parameters
        ----------
        epochs : int
            number of epochs (iterations per sample)
        minibatch : int
            minibatch size
        monitor_every : int
            monitoring frequency
        stop_on_nan : bool (default: False)
            if True, will stop if loss becomes NaN
        tol : float
            tolerance criterion for stopping based on training set
        verbose : bool
            if True, print progress during training

        Returns
        -------
        dict : containing loss values and possibly additional keys
        """

        # initialize variables
        iter = 0

        # minibatch size
        minibatch = self.n_trn_data if minibatch is None else minibatch
        if minibatch > self.n_trn_data:
            minibatch = self.n_trn_data

        maxiter = int(self.n_trn_data / minibatch + 0.5) * epochs 
            
        # placeholders for outputs
        trn_outputs = {}
        for key in self.trn_outputs_names:
            trn_outputs[key] = []

        # cast trn_data
        self.trn_data = [x.astype(dtype) for x in self.trn_data]

        if not verbose:
            pbar = no_tqdm()
        else:
            pbar = progressbar(total=maxiter * minibatch)
            desc = 'Training '
            if type(verbose) == str:
                desc += verbose
            pbar.set_description(desc)

        with pbar:
            # loop over epochs
            for epoch in range(epochs):
                # set learning rate
                lr_epoch = self.lr * (self.lr_decay**epoch)
                self.lr_op.set_value(lr_epoch)

                # loop over batches
                for trn_batch in iterate_minibatches(self.network, self.trn_data, minibatch,
                                                     seed=self.gen_newseed()):
                    trn_batch = tuple(trn_batch)

                    outputs = self.make_update(*trn_batch)

                    for name, value in zip(self.trn_outputs_names, outputs):
                        trn_outputs[name].append(value)

                    trn_loss = trn_outputs['loss'][-1]
                    diff = self.loss - trn_loss
                    self.loss = trn_loss

                    # check for convergence
                    if tol is not None:
                        if abs(diff) < tol:
                            break

                    # check for nan
                    if stop_on_nan and np.isnan(trn_loss):
                        break

                    pbar.update(minibatch)

        # convert lists to arrays
        for name, value in trn_outputs.items():
            trn_outputs[name] = np.asarray(value)

        return trn_outputs

    def gen_newseed(self):
        """Generates a new random seed"""
        if self.seed is None:
            return None
        else:
            return self.rng.randint(0, 2**31)


def iterate_minibatches(network, trn_data, minibatch=10, seed=None):
    """Minibatch iterator

    Parameters
    ----------
    trn_data : tuple of arrays
        Training daa
    minibatch : int
        Size of batches
    seed : None or int
        Seed for minibatch order

    Returns
    -------
    trn_batch : tuple of arrays
        Batch of training data
    """
    n_samples = len(trn_data[0])
    indices = np.arange(n_samples)

    rng = np.random.RandomState(seed=seed)
    rng.shuffle(indices)

    weight_Z = np.sum(trn_data[2])

    start_idx = 0

    queue = None
    batches = []
    while start_idx < n_samples:
        total_weight = 0
        end_idx = start_idx

        excerpt = [[], [], []]

        while True:
            if queue == None:
                if end_idx >= n_samples:
                    break

                queue = [ td[indices[end_idx]] for td in trn_data ] 
                queue[2] = queue[2] * n_samples / weight_Z
                end_idx += 1

            if queue[2] == 0:
                queue = None
                continue

            if total_weight + queue[2] <= minibatch:
                total_weight += queue[2]

                for i in range(3):
                    excerpt[i].append(np.asarray(queue[i]))
                queue = None
            else:
                new_excerpt = [ q for q in queue ]
                new_excerpt[2] = minibatch - total_weight

                if new_excerpt[2] != 0:
                    for i in range(3):
                        excerpt[i].append(np.asarray(new_excerpt[i]))

                queue[2] -= minibatch - total_weight
                total_weight = minibatch
                break
            
        batches.append(excerpt)
        start_idx = end_idx

    rng.shuffle(batches)

    return batches
