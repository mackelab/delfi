import lasagne.updates as lu
import numpy as np
import theano
import theano.tensor as tt
import sys

from delfi.utils.progress import no_tqdm, progressbar
from numpy.lib.stride_tricks import as_strided

dtype = theano.config.floatX


def block_circulant(x):
    
    n, d = x.shape
    b = x.itemsize
    
    y = as_strided(np.tile(x.flatten(), 2),
                   shape=(n, n, d),
                   strides=(d * b, d * b, b),
                   writeable=False)
    return y.reshape(n**2, d)


class Trainer:
    def __init__(self, network, loss, trn_data, trn_inputs, step=lu.adam, lr=0.001, lr_decay=1.0, max_norm=0.1,
                 monitor=None, val_frac=0.0, assemble_extra_inputs=None, seed=None):
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
        val_frac: float
            Fraction of dataset to use as validation set
        assemble_extra_inputs: function
            (optional) function to compute extra inputs needed to evaluate loss
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

        self.assemble_extra_inputs = assemble_extra_inputs

        self.do_validation = val_frac > 0

        if self.do_validation:

            n_trn = int((1 - val_frac) * self.n_trn_data)
            self.val_data = [data[n_trn:] for data in trn_data].copy()  # copy() might be overly prudent
            self.trn_data = [data[:n_trn] for data in trn_data].copy()

            # compile theano function for validation
            self.eval_loss = theano.function(inputs=self.trn_inputs, outputs=self.loss)
            self.best_val_loss = np.inf

        # initialize variables
        self.loss = float('inf')

    def calc_validation_loss(self, minibatch, strict_batch_size):
        s, L, n_val, n_batches_used = 0, 0.0, self.val_data[0].shape[0], 0.0  # n_batches_used can be fractional

        while s < n_val and (not strict_batch_size or s + minibatch <= n_val):
            e = np.minimum(s + minibatch, n_val)
            next_batch = tuple([x[s:e] for x in self.trn_data])
            if self.assemble_extra_inputs is not None:
                next_batch = self.assemble_extra_inputs(next_batch)
            L += self.eval_loss(*next_batch) * (s - e) / minibatch
            n_batches_used += (s - e) / minibatch
            s += minibatch

        return L / n_batches_used

    def train(self,
              epochs=250,
              minibatch=50,
              patience=20,
              monitor_every=None,
              stop_on_nan=False,
              strict_batch_size=False,
              tol=None,
              verbose=False,
              print_each_epoch=False):
        """Trains the model

        Parameters
        ----------
        epochs : int
            number of epochs (iterations per sample)
        minibatch : int
            minibatch size
        monitor_every : float
            after how many epochs should validation loss be checked?
        stop_on_nan : bool (default: False)
            if True, will stop if loss becomes NaN
        tol : float
            tolerance criterion for stopping based on training set
        verbose : bool
            if True, print progress during training
        strict_batch_size : bool
            Whether to ignore last batch if it would be smaller than minibatch
        print_each_epoch: bool
            Whether to print a period `.' each epoch, useful to avoid timeouts in continuous integration.

        Returns
        -------
        dict : containing loss values and possibly additional keys
        """

        # initialize variables
        iter = 0
        patience_left = patience        
        if monitor_every is None:
            monitor_every = min(10 ** 5 / float(self.n_trn_data), 1.0)
        logger = sys.stdout

        # minibatch size
        minibatch = self.n_trn_data if minibatch is None else minibatch
        if minibatch > self.n_trn_data:
            minibatch = self.n_trn_data

        if self.do_validation and strict_batch_size:
            assert self.val_data[0].shape[0] >= minibatch, "not enough validation samples for a minibatch"
            if self.val_data[0].shape[0] % minibatch != 0 and verbose:
                print('{0} validation samples not a multiple of minibatch size {1}, some samples will be wasted'.
                      format(self.val_data[0].shape[0], minibatch))

        maxiter = int(self.n_trn_data / minibatch + 0.5) * epochs

        # placeholders for outputs
        trn_outputs = {}
        for key in self.trn_outputs_names:
            trn_outputs[key] = []

        if self.do_validation:
            trn_outputs['val_loss'], trn_outputs['val_loss_iter'] = [], []

        # cast trn_data
        self.trn_data = [x.astype(dtype) for x in self.trn_data]

        if not verbose:
            pbar = no_tqdm()
        else:
            pbar = progressbar(total=maxiter * minibatch)
            desc = 'Training on {0} samples'.format(self.trn_data[0].shape[0])
            if type(verbose) == str:
                desc += verbose
            pbar.set_description(desc)

        break_flag = False
        with pbar:
            # loop over epochs
            for epoch in range(epochs):
                # set learning rate
                lr_epoch = self.lr * (self.lr_decay**epoch)
                self.lr_op.set_value(lr_epoch)

                # loop over batches
                for trn_batch in iterate_minibatches(self.trn_data, minibatch,
                                                     seed=self.gen_newseed(),
                                                     strict_batch_size=strict_batch_size):

                    if self.assemble_extra_inputs is not None:
                        trn_batch = self.assemble_extra_inputs(tuple(trn_batch))
                    else: 
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
                            break_flag = True
                            break

                    # check for nan
                    if stop_on_nan and np.isnan(trn_loss):
                        print('stopping due to NaN value on iteration {0}\n'.format(iter))
                        break_flag = True
                        break

                    # validation-data tracking of convergence
                    if self.do_validation:
                        epoch_frac = (iter * minibatch) / self.n_trn_data  # how many epochs so far
                        prev_epoch_frac = ((iter - 1) * minibatch) / self.n_trn_data
                        # do validation if we've passed a multiple of monitor_every epochs
                        if iter == 0 or \
                                np.floor(epoch_frac / monitor_every) != np.floor(prev_epoch_frac / monitor_every):
                            val_loss = self.calc_validation_loss(minibatch, strict_batch_size)
                            trn_outputs['val_loss'].append(val_loss)
                            trn_outputs['val_loss_iter'].append(iter)
                            patience_left -= 1

                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                patience_left = patience  # reset patience_left

                            if patience_left <= 0:
                                break_flag = True
                                if verbose:
                                    print('Stopping at epoch {0}, '
                                          'training loss = {1}, '
                                          'validation loss = {2}\n'
                                          .format(epoch_frac, trn_loss, val_loss))
                                break
                    pbar.update(minibatch)
                    iter += 1
                if print_each_epoch:
                    print('.')
                if break_flag:
                    break

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


def iterate_minibatches(trn_data, minibatch=10, seed=None, strict_batch_size=False):
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

    start_idx = 0

    for start_idx in range(0, n_samples-minibatch+1, minibatch):
        excerpt = indices[start_idx:start_idx + minibatch]

        yield (trn_data[k][excerpt] for k in range(len(trn_data)))

    rem_i = n_samples - (n_samples % minibatch)
    if not strict_batch_size and rem_i != n_samples:
        excerpt = indices[rem_i:]
        yield (trn_data[k][excerpt] for k in range(len(trn_data)))


class ActiveTrainer(Trainer):

    def __init__(self, network, loss, trn_data, trn_inputs,
                 step=lu.adam, lr=0.001, lr_decay=1.0, max_norm=0.1,
                 monitor=None, val_frac=0., seed=None,               
                 generator=None, n_atoms=1, moo='resample', obs=None):
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
        val_frac: float
            Fraction of dataset to use as validation set
        seed : int or None
            If provided, random number generator for batches will be seeded
        generator: delfi generator object
            Object with gen method to generate additional simulation parameters
        n_atoms: int
            Number of additional simulation parameters to draw for atomic APT
        moo: string
            Mode of operation for generation of additional simulation parameters
        obs: ndarray
            Observed summary statistics 
        """
        if network.density == 'maf':
            f_assemble_extra_inputs = self.assemble_extra_inputs_maf
        elif network.density == 'mog':
            f_assemble_extra_inputs = self.assemble_extra_inputs_mdn
        else:
            raise NotImplementedError

        def assemble_extra_inputs(trn_data):
            return f_assemble_extra_inputs(
                    trn_data=trn_data, 
                    generator=generator,
                    n_atoms=n_atoms,
                    moo=moo,
                    obs=obs)

        super().__init__(network=network, loss=loss, 
            trn_data=trn_data, trn_inputs=trn_inputs,
            step=step, lr=lr,lr_decay=lr_decay, max_norm=max_norm,
            monitor=monitor, val_frac=val_frac, seed=seed,
            assemble_extra_inputs=assemble_extra_inputs)

    def assemble_extra_inputs_mdn(self, trn_data, generator, n_atoms,
                                  moo='resample', obs=None):
        """convenience function for assembling input for network training"""

        batchsize = trn_data[0].shape[0]
        if moo == 'resample':
            # all-to-all comparison of theta's and x's, n_atoms loss evaluations

            th_nl = np.empty((batchsize, n_atoms, trn_data[0].shape[1]), dtype=dtype)
            for n in range(batchsize):
                idx = self.rng.choice(batchsize-1, n_atoms, replace=False) + n + 1
                th_nl[n, :, :] = trn_data[0][np.mod(idx, batchsize)]
            theta_all = np.concatenate(
                (trn_data[0].reshape((batchsize, 1, -1)),
                 th_nl), axis=1)
        else:
            raise NotImplemented('mode of operation not supported')

        # compute log prior ratios (assuming atomic proposal)
        lprs = generator.prior.eval(
            theta_all.reshape(batchsize * (n_atoms + 1), -1),
            log=True).reshape(batchsize, n_atoms + 1).astype(dtype)

        # theta_all : (n_batch * (n_atoms + 1)  x n_outputs
        # lprs  : n_batch x (n_atoms+1)
        # x : n_batch x n_inputs
        trn_data = (theta_all, trn_data[1], lprs, *trn_data[2:])

        return trn_data

    def assemble_extra_inputs_maf(self, trn_data, generator, n_atoms,
                                  moo='resample', obs=None):
        """convenience function for assembling input for network training.
        Note the data are assembled in a different ordering than for the MDN.
        """
        batchsize = trn_data[0].shape[0]

        if moo == 'resample':

            if n_atoms < batchsize - 1:
                # all-to-all comparison of theta's and x's, n_atoms loss evaluations

                idx_ = np.empty((batchsize, n_atoms), dtype=int)
                for n in range(batchsize):
                    idx_[n] = self.rng.choice(batchsize - 1, n_atoms,
                                              replace=False)
                idx_ = (idx_ + np.arange(1, batchsize + 1).reshape(-1, 1)).reshape(-1)
                th_nl = np.vstack((trn_data[0], trn_data[0][np.mod(idx_, batchsize)]))

            else: 

                assert n_atoms < batchsize
                th_nl = block_circulant(trn_data[0])

        else:
            raise NotImplemented('mode of operation not supported')

        # compute log prior ratios (assuming atomic proposal)
        lprs = generator.prior.eval(th_nl, log=True).reshape(n_atoms + 1, batchsize).astype(dtype)

        # inputs to MAF (summary stats)
        x_nl = np.tile(trn_data[1], (n_atoms + 1, 1))

        # th_nl : (n_batch * (n_atoms + 1), n_outputs)
        # lprs  : ((n_atoms + 1) * n_batch)
        # x_nl  : ((n_batch * (n_atoms + 1), *input_size)
        return (th_nl, x_nl, lprs, *trn_data[2:])
