import numpy as np
from delfi.neuralnet.loss.lossfunc import snpe_loss_prior_as_proposal
from delfi.neuralnet.Trainer import Trainer


def normalize_cmaf(cmaf, f_accept, xs, n_samples=10, seed=None, val_frac=0.1,
                   minibatch=100, epochs=200, verbose=False, stop_on_nan=False):
    from snl.ml.models.mafs import ConditionalMaskedAutoregressiveFlow
    assert isinstance(cmaf, ConditionalMaskedAutoregressiveFlow)

    # first, sample from the existing MAF and apply our acceptance criterion
    xs = np.repeat(xs, n_samples, axis=0)
    thetas = np.full(xs.shape, np.nan)
    rng = np.random.randomstate(seed=seed)

    jj = 0  # index into final array of thetas
    for i, x in enumerate(xs):
        n_accepted = 0
        while n_accepted < n_samples:
            next_thetas = \
                cmaf.gen(x=x, n_samples=n_samples - n_accepted, rng=rng)
            for theta in next_thetas:
                if not f_accept(theta):
                    continue
                thetas[jj, :] = theta
                n_accepted += 1
                jj += 1

    # create a new MAF
    newrng = np.random.RandomState(seed=seed)
    cmaf_new = ConditionalMaskedAutoregressiveFlow(
        n_inputs=cmaf.n_inputs,
        n_outputs=cmaf.n_outputs,
        n_hiddens=cmaf.n_hiddens,
        act_fun=cmaf.act_fun,
        n_mades=cmaf.n_mades,
        batch_norm=cmaf.batch_norm,
        mode=cmaf.mode,
        input=None,  # is this ok?
        output=None,  # is this ok?
        rng=newrng,
        output_order=cmaf.output_order)  # hope this is ok?

    # train network by directly maximizing q(\theta | x)
    loss, trn_inputs = snpe_loss_prior_as_proposal(cmaf, svi=False)
    trn_data = (thetas, xs)

    t = Trainer(network=cmaf, loss=loss, trn_data=trn_data,
                trn_inputs=trn_inputs, seed=seed + 5)
    log = t.train(epochs=epochs, minibatch=minibatch,
                  verbose=verbose, stop_on_nan=stop_on_nan, val_frac=val_frac)
