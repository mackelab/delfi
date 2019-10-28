import theano
import theano.tensor as tt
import delfi.distribution as dd
import numpy as np
from delfi.neuralnet.NeuralNet import NeuralNet, dtype
from delfi.utils.symbolic import tensorN, mog_LL, MyLogSumExp, invert_each, det_each, cholesky_each


def snpe_loss_prior_as_proposal(model, svi=False):
    """
    Simplest loss function, for the case where the prior is the same as the
    proposal. In this case snpe-A, snpe-B and APT are equivalent, and we're
    simply maximizing the likelihood of params under the parameterized mdn.

    Parameters
    ----------
    model : NeuralNet or ConditionalMaskedAutoregressiveFlow
        Mixture density network or conditional masked autoregressive flow.
    svi : bool
        Whether to use SVI version of the model or not (only for MDN models)

    Returns
    -------
    loss : theano scalar
        Loss function
    trn_inputs : list
        Tensors to be provided to the loss function during training
    """
    # note that lprobs and dlprobs are the same for non-svi networks
    loss = -tt.mean(model.lprobs) if svi else -tt.mean(model.dlprobs)
    trn_inputs = [model.params, model.stats]
    return loss, trn_inputs


def snpeb_loss(model, svi=False):
    # note that lprobs and dlprobs are the same for non-svi networks
    lprobs = model.lprobs if svi else model.dlprobs
    iws = tensorN(1, name='iws', dtype=dtype)  # importance weights
    loss = -tt.mean(iws * lprobs)
    # collect extra input variables to be provided for each training data point
    trn_inputs = [model.params, model.stats, iws]
    return loss, trn_inputs


def apt_loss_MoG_proposal(mdn, prior, n_proposal_components=None, svi=False, add_prior_precision=True, Ptol=1e-7):
    """Define loss function for training with a MoG proposal, allowing
    the proposal distribution to be different for each sample. The proposal
    means, precisions and weights are passed along with the stats and params.

    This function computes a symbolic expression but does not actually compile
    or calculate that expression.

    This loss does not include any regularization or prior terms, which must be
    added separately before compiling.

    Parameters
    ----------
    mdn : NeuralNet
        Mixture density network.
    svi : bool
        Whether to use SVI version of the mdn or not

    Returns
    -------
    loss : theano scalar
        Loss function
    trn_inputs : list
        Tensors to be provided to the loss function during training
    """
    assert mdn.density == 'mog'
    uniform_prior = isinstance(prior, dd.Uniform)
    if not uniform_prior and not isinstance(prior, dd.Gaussian):
        raise NotImplemented  # prior must be Gaussian or uniform
    ncprop = mdn.n_components if n_proposal_components is None \
        else n_proposal_components  # default to component count of posterior

    nbatch = mdn.params.shape[0]

    # a mixture weights, ms means, P=U^T U precisions, QFs are tensorQF(P, m)
    a, ms, Us, ldetUs, Ps, ldetPs, Pms, QFs = \
        mdn.get_mog_tensors(return_extras=True, svi=svi)
    # convert mixture vars from lists of tensors to single tensors, of sizes:
    las = tt.log(a)
    Us = tt.stack(Us, axis=3).dimshuffle(0, 3, 1, 2)
    Ps = tt.stack(Ps, axis=3).dimshuffle(0, 3, 1, 2)
    Pms = tt.stack(Pms, axis=2).dimshuffle(0, 2, 1)
    ldetPs = tt.stack(ldetPs, axis=1)
    QFs = tt.stack(QFs, axis=1)
    # as: (batch, mdn.n_components)
    # Ps: (batch, mdn.n_components, n_outputs, n_outputs)
    # Pm: (batch, mdn.n_components, n_outputs)
    # ldetPs: (batch, mdn.n_components)
    # QFs: (batch, mdn.n_components)

    # Define symbolic variables, that hold for each sample's MoG proposal:
    # precisions times means (batch, ncprop, n_outputs)
    # precisions (batch, ncprop, n_outputs, n_outputs)
    # log determinants of precisions (batch, ncprop)
    # log mixture weights (batch, ncprop)
    # quadratic forms QF = m^T P m (batch, ncprop)
    prop_Pms = tensorN(3, name='prop_Pms', dtype=dtype)
    prop_Ps = tensorN(4, name='prop_Ps', dtype=dtype)
    prop_ldetPs = tensorN(2, name='prop_ldetPs', dtype=dtype)
    prop_las = tensorN(2, name='prop_las', dtype=dtype)
    prop_QFs = tensorN(2, name='prop_QFs', dtype=dtype)

    # calculate corrections to precisions (P_0s) and precisions * means (Pm_0s)
    P_0s = prop_Ps
    Pm_0s = prop_Pms
    if not uniform_prior and not add_prior_precision:  # Gaussian prior
        P_0s = P_0s - prior.P
        Pm_0s = Pm_0s - prior.Pm

    # To calculate the proposal posterior, we multiply all mixture component
    # pdfs from the true posterior by those from the proposal. The resulting
    # new mixture is ordered such that all product terms involving the first
    # component of the true posterior appear first. The shape of pp_Ps is
    # (batch, mdn.n_components, ncprop, n_outputs, n_outputs)
    pp_Ps = Ps.dimshuffle(0, 1, 'x', 2, 3) + P_0s.dimshuffle(0, 'x', 1, 2, 3)
    spp = pp_Ps.shape

    # get square roots of diagonal entries of posterior proposal precision components, which are equal to the L2 norms
    # of the Cholesky factor columns for the same matrix. we'll use these to improve the numerical conditioning of pp_Ps
    ds = tt.sqrt(tt.sum(pp_Ps * np.eye(mdn.n_outputs), axis=4))
    Ds = ds.dimshuffle(0, 1, 2, 'x', 3) * ds.dimshuffle(0, 1, 2, 3, 'x')
    # normalize the estimate of each true posterior component according to the corresponding elements of d:
    # first normalize the Cholesky factor of the true posterior component estimate, separately for each pairing of
    # proposal and true posterior components...
    Us_normed = Us.dimshuffle(0, 1, 'x', 2, 3) / ds.dimshuffle(0, 1, 2, 'x', 3)
    # then normalize the propsal. the resulting list is the same proposal, differently normalized for each component of
    # the true posterior
    P_0s_normed = P_0s.dimshuffle(0, 'x', 1, 2, 3) / Ds

    # reshape temporarily so there's just one batch dimension
    Us_normed_R = Us_normed.reshape((-1, mdn.n_outputs, mdn.n_outputs))
    Ps_normed_R = tt.batched_dot(Us_normed_R.dimshuffle(0, 2, 1), Us_normed_R)
    Ps_normed = Ps_normed_R.reshape(spp)

    pp_Ps_normed = Ps_normed + P_0s_normed + np.eye(mdn.n_outputs) * Ptol
    # lower Cholesky factors for normalized precisions of proposal posterior components
    pp_Ls_normed = cholesky_each(pp_Ps_normed)
    # log determinants of lower Cholesky factors for normalized precisions of proposal posterior components
    pp_ldetLs_normed = tt.sum(tt.log(tt.sum(pp_Ls_normed * np.eye(mdn.n_outputs), axis=4)), axis=3)
    # precisions of proposal posterior components (now well-conditioned)
    pp_Ps = pp_Ps_normed * Ds
    # log determinants of proposal posterior precisions
    pp_ldetPs = 2.0 * (tt.sum(tt.log(ds), axis=3) + pp_ldetLs_normed)
    # covariance matrices
    pp_Ss = invert_each(pp_Ps)  # covariances of proposal posterior components
    # precision times mean for each proposal posterior component:
    pp_Pms = Pms.dimshuffle(0, 1, 'x', 2) + Pm_0s.dimshuffle(0, 'x', 1, 2)
    # mean of proposal posterior components:
    pp_ms = (pp_Ss * pp_Pms.dimshuffle(0, 1, 2, 'x', 3)).sum(axis=4)
    # quadratic form defined by each pp_P evaluated at each pp_m
    pp_QFs = (pp_Pms * pp_ms).sum(axis=3)

    # normalization constants for integrals of Gaussian product-quotients
    # (for Gaussian proposals) or Gaussian products (for uniform priors)
    # Note we drop a "constant" (for each combination of sample, proposal
    # component posterior component) term of
    #
    # 0.5 * (tensorQF(prior.P, prior.m) - prior_ldetP)
    #
    # since we're going to normalize the pp mixture coefficients sum to 1
    pp_lZs = 0.5 * ((ldetPs - QFs).dimshuffle(0, 1, 'x') +
                    (prop_ldetPs - prop_QFs).dimshuffle(0, 'x', 1) -
                    (pp_ldetPs - pp_QFs))

    # calculate non-normalized log mixture coefficients of the proposal
    # posterior by adding log posterior weights a to normalization coefficients
    # Z. These do not yet sum to 1 in the linear domain
    pp_las_nonnormed = \
        las.dimshuffle(0, 1, 'x') + prop_las.dimshuffle(0, 'x', 1) + pp_lZs

    # reshape tensors describing proposal posterior components so that there's
    # only one dimension that ranges over components
    ncpp = ncprop * mdn.n_components  # number of proposal posterior components
    pp_las_nonnormed = pp_las_nonnormed.reshape((nbatch, ncpp))
    pp_ldetPs = pp_ldetPs.reshape((nbatch, ncpp))
    pp_ms = pp_ms.reshape((nbatch, ncpp, mdn.n_outputs))
    pp_Ps = pp_Ps.reshape((nbatch, ncpp, mdn.n_outputs, mdn.n_outputs))

    # normalize log mixture weights so they sum to 1 in the linear domain
    pp_las = pp_las_nonnormed - MyLogSumExp(pp_las_nonnormed, axis=1)

    mog_LL_inputs = \
        [(pp_ms[:, i, :], pp_Ps[:, i, :, :], pp_ldetPs[:, i])
         for i in range(ncpp)]  # list (over comps) of tuples (over vars)
    # 2 tensor inputs, lists (over comps) of tensors:
    mog_LL_inputs = [mdn.params, pp_las, *zip(*mog_LL_inputs)]

    loss = -tt.mean(mog_LL(*mog_LL_inputs))

    # collect extra input variables to be provided for each training data point
    trn_inputs = [mdn.params, mdn.stats,
                  prop_Pms, prop_Ps, prop_ldetPs, prop_las, prop_QFs]

    return loss, trn_inputs


def apt_loss_gaussian_proposal(mdn, prior, svi=False, add_prior_precision=True, Ptol=1e-7):
    """Define loss function for training with a Gaussian proposal, allowing
    the proposal distribution to be different for each sample. The proposal
    mean and precision are passed along with the stats and params.

    This function computes a symbolic expression but does not actually compile
    or calculate that expression.

    This loss does not include any regularization or prior terms, which must be
    added separately before compiling.

    Parameters
    ----------
    mdn : NeuralNet
        Mixture density network.
    svi : bool
        Whether to use SVI version of the mdn or not

    Returns
    -------
    loss : theano scalar
        Loss function
    trn_inputs : list
        Tensors to be provided to the loss function during training
    prior: delfi distribution
        Prior distribution on parameters
    """
    assert mdn.density == 'mog'
    uniform_prior = isinstance(prior, dd.Uniform)
    if not uniform_prior and not isinstance(prior, dd.Gaussian):
        raise NotImplemented  # prior must be Gaussian or uniform

    # a mixture weights, ms means, P=U^T U precisions, QFs are tensorQF(P, m)
    a, ms, Us, ldetUs, Ps, ldetPs, Pms, QFs = \
        mdn.get_mog_tensors(return_extras=True, svi=svi)

    # define symbolic variables to hold for each sample's Gaussian proposal:
    # means (batch, n_outputs)
    # precisions (batch, n_outputs, n_outputs)
    prop_m = tensorN(2, name='prop_m', dtype=dtype)
    prop_P = tensorN(3, name='prop_P', dtype=dtype)

    # calculate corrections to precision (P_0) and precision * mean (Pm_0)
    P_0 = prop_P
    Pm_0 = tt.sum(prop_P * prop_m.dimshuffle(0, 'x', 1), axis=2)
    if not uniform_prior and not add_prior_precision:  # Gaussian prior
        P_0 = P_0 - prior.P
        Pm_0 = Pm_0 - prior.Pm

    # precisions of proposal posterior components (before numerical conditioning step)
    pp_Ps = [P + P_0 for P in Ps]

    # get square roots of diagonal entries of posterior proposal precision components, which are equal to the L2 norms
    # of the Cholesky factor columns for the same matrix. we'll use these to improve the numerical conditioning of pp_Ps
    ds = [tt.sqrt(tt.sum(pp_P * np.eye(mdn.n_outputs), axis=2)) for pp_P in pp_Ps]
    # normalize the estimate of each true posterior component according to the corresponding elements of d:
    # first normalize the Cholesky factor of the true posterior component estimate...
    Us_normed = [U / d.dimshuffle(0, 'x', 1) for U, d in zip(Us, ds)]
    # then normalize the propsal. the resulting list is the same proposal, differently normalized for each component of
    # the true posterior
    P_0s_normed = [P_0 / (d.dimshuffle(0, 'x', 1) * d.dimshuffle(0, 1, 'x')) for d in ds]
    pp_Ps_normed = [tt.batched_dot(U_normed.dimshuffle(0, 2, 1), U_normed) + P_0_normed + np.eye(mdn.n_outputs) * Ptol
                    for U_normed, P_0_normed in zip(Us_normed, P_0s_normed)]
    # lower Cholesky factors for normalized precisions of proposal posterior components
    pp_Ls_normed = [cholesky_each(pp_P_normed) for pp_P_normed in pp_Ps_normed]
    # log determinants of lower Cholesky factors for normalized precisions of proposal posterior components
    pp_ldetLs_normed = [tt.sum(tt.log(tt.sum(pp_L_normed * np.eye(mdn.n_outputs), axis=2)), axis=1)
                        for pp_L_normed in pp_Ls_normed]
    # precisions of proposal posterior components (now well-conditioned)
    pp_Ps = [d.dimshuffle(0, 1, 'x') * pp_P_normed * d.dimshuffle(0, 'x', 1)
             for pp_P_normed, d in zip(pp_Ps_normed, ds)]
    # log determinants of proposal posterior precisions
    pp_ldetPs = [2.0 * (tt.sum(tt.log(d), axis=1) + pp_ldetL_normed)
                 for d, pp_ldetL_normed in zip(ds, pp_ldetLs_normed)]

    # covariances of proposal posterior components:
    pp_Ss = [invert_each(P) for P in pp_Ps]
    # precision times mean for each proposal posterior component:
    pp_Pms = [Pm + Pm_0 for Pm in Pms]
    # mean of proposal posterior components:
    pp_ms = [tt.batched_dot(S, Pm) for S, Pm in zip(pp_Ss, pp_Pms)]
    # quadratic form defined by each pp_P evaluated at each pp_m
    pp_QFs = [tt.sum(m * Pm, axis=1) for m, Pm in zip(pp_ms, pp_Pms)]

    # normalization constants for integrals of Gaussian product-quotients
    # (for Gaussian proposals) or Gaussian products (for uniform priors)
    # Note we drop a "constant" (for each sample, w.r.t trained params) term of
    #
    # 0.5 * (prop_ldetP - prior_ldetP + tensorQF(prior.P, prior.m) - tensorQF(prop_P, prop_m))
    #
    # since we're going to normalize the pp mixture coefficients sum to 1
    pp_lZs = [0.5 * (ldetP - pp_ldetP - QF + pp_QF)
              for    ldetP,  pp_ldetP,  QF,  pp_QF
              in zip(ldetPs, pp_ldetPs, QFs, pp_QFs)]

    # calculate log mixture coefficients of proposal posterior in two steps:
    # 1) add log posterior weights a to normalization coefficients Z
    # 2) normalize to sum to 1 in the linear domain, but stay in the log domain
    pp_las = tt.stack(pp_lZs, axis=1) + tt.log(a)
    pp_las = pp_las - MyLogSumExp(pp_las, axis=1)

    loss = -tt.mean(mog_LL(mdn.params, pp_las, pp_ms, pp_Ps, pp_ldetPs))

    # collect extra input variables to be provided for each training data point
    trn_inputs = [mdn.params, mdn.stats, prop_m, prop_P]

    return loss, trn_inputs


def apt_loss_atomic_proposal(model, svi=False, combined_loss=False):
    """Define loss function for training with a atomic proposal. Assumes a
    uniform proposal distribution over each sample parameter and an externally
    provided set of alternatives.

    model : NeuralNet or ConditionalMaskedAutoregressiveFlow
        Mixture density network or conditional masked autoregressive flow.
    svi : bool
        Whether to use SVI version of the mdn or not
    """

    if model.density == 'mog':
        return apt_mdn_loss_atomic_proposal(model, svi=svi, combined_loss=combined_loss)
    elif model.density == 'maf':
        assert not svi, 'SVI not supported for MAFs'
        return apt_maf_loss_atomic_proposal(model, svi=svi, combined_loss=combined_loss)


def apt_mdn_loss_atomic_proposal(mdn, svi=False, combined_loss=False):
    """Define loss function for training with a atomic proposal. Assumes a
    uniform proposal distribution over each sample parameter and an externally
    provided set of alternatives.

    mdn: NeuralNet
        Mixture density network.
    svi : bool
        Whether to use SVI version of the mdn or not
    """
    assert mdn.density == 'mog'

    # a is mixture weights, ms are means, U^T U are precisions
    a, ms, Us, ldetUs = mdn.get_mog_tensors(svi=svi)

    # define symbolic variable to hold params that will be inferred
    # theta_all : (n_batch * (n_atoms + 1)  x n_outputs
    # lprs  : n_batch x (n_atoms+1)
    theta_all = tensorN(3, name='params_nl', dtype=dtype)  # true (row 1), atoms
    lprs = tensorN(2, name='lprs', dtype=dtype)  # log tilde_p / p

    # calculate Mahalanobis distances distances wrt U'U for every theta,x pair
    # diffs : [ n_batch x (n_atoms+1) x n_outputs for each component ]
    # Ms    : [ n_batch x (n_atoms+1)             for each component ]
    # Ms[k][n,i] = (theta[i] - m[k][n])' U[k][n]' U[k][n] (theta[i] - m[k][n])
    dthetas = [theta_all - m.dimshuffle([ 0,'x',1]) for m in ms] # theta[i] - m[k][n]
    Ms = [tt.sum( tt.sum(dtheta.dimshuffle([0,1,'x',2])*U.dimshuffle([0,'x',1,2]),
        axis=3)**2, axis=2 ) for dtheta,U in zip(dthetas, Us)]

    # compute (unnormalized) log-densities, weighted by log prior ratios
    Ms = [-0.5 * M - lprs for M in Ms]

    # compute per-component log-densities and log-normalizers
    lprobs_comps = [M[:,0] + ldetU for M, ldetU in zip(Ms, ldetUs)]
    lZ_comps = [MyLogSumExp(M,axis=1).squeeze() + ldetU
            for M,ldetU in zip(Ms, ldetUs)]  # sum over all proposal thetas

    # compute overall log-densities and log-normalizers across components
    lq = MyLogSumExp(tt.stack(lprobs_comps, axis=1) + tt.log(a), axis=1)
    lZ = MyLogSumExp(tt.stack(lZ_comps, axis=1) + tt.log(a), axis=1)

    lprobs = lq.squeeze() - lZ.squeeze()

    # collect the extra input variables that have to be provided for each
    # training data point
    trn_inputs = [theta_all, mdn.stats, lprs]
    if combined_loss:  # add prior loss on prior samples
        l_ml = lq.squeeze()  # direct posterior evalution
        is_prior_sample = tensorN(1, name='prop_mask', dtype=dtype)
        trn_inputs.append(is_prior_sample)
        loss = -tt.mean(lprobs + is_prior_sample * l_ml)
    else:
        loss = -tt.mean(lprobs)  # average over samples

    return loss, trn_inputs    


def apt_maf_loss_atomic_proposal(net, svi=False, combined_loss=False):
    """Define loss function for training with a atomic proposal. Assumes a
    uniform proposal distribution over each sample parameter and an externally
    provided set of alternatives.

    net: MAF-based conditional density net
    svi : bool
        Whether to use SVI version of the mdn or not
    """
    assert net.density == 'maf'
    assert not svi, 'SVI not supported for MAFs'

    # define symbolic variable to hold params that will be inferred
    # params : n_batch x  n_outputs
    # all_thetas : (n_batch * (n_atoms + 1)  x n_outputs
    # lprs  : (n_atoms + 1) x n_batch
    # stats :  n_batch x  n_inputs
    # x_nl  : (n_batch * (n_atoms + 1)) x n_inputs
    theta_all = tensorN(2, name='params_nl', dtype=dtype)
    x_nl = tensorN(2, name='stats_nl', dtype=dtype)
    lprs = tensorN(2, name='lprs', dtype=dtype)  # log tilde_p / p

    n_batch = tt.shape(lprs)[1]
    n_atoms = tt.shape(lprs)[0] - 1

    # compute MAF log-densities for true and other atoms
    lprobs = theano.clone(output=net.lprobs,
                          replace={net.params:theta_all, net.stats:x_nl},
                          share_inputs=True)
    lprobs = tt.reshape(lprobs, newshape=(n_atoms + 1, n_batch), ndim=2)

    # compute nonnormalized log posterior probabilities
    atomic_ppZ = lprobs - lprs
    # compute posterior probability of true params in atomic task
    atomic_pp = atomic_ppZ[0, :].squeeze() - \
        MyLogSumExp(atomic_ppZ, axis=0).squeeze()

    # collect the extra input variables that have to be provided for each
    # training data point, and calculate the loss by averaging over samples
    trn_inputs = [theta_all, x_nl, lprs]
    if combined_loss:  # add prior loss on prior samples
        l_ml = lprobs[0, :].squeeze()  # direct posterior evaluation
        is_prior_sample = tensorN(1, name='prop_mask', dtype=dtype)
        trn_inputs.append(is_prior_sample)
        loss = -tt.mean(atomic_pp + is_prior_sample * l_ml)
    else:
        loss = -tt.mean(atomic_pp)

    return loss, trn_inputs
