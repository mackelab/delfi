import delfi.distribution as dd
import delfi.generator as dg
import delfi.inference as infer
import delfi.summarystats as ds
import numpy as np
import theano

from delfi.simulator.Gauss import Gauss
from delfi.neuralnet.NeuralNet import MAFconditional


def simplegaussprod(m1, S1, m2, S2):
    m1 = m1.squeeze()
    m2 = m2.squeeze()
    P1 = np.linalg.inv(S1)
    Pm1 = np.dot(P1, m1)
    P2 = np.linalg.inv(S2)
    Pm2 = np.dot(P2, m2)
    P = P1 + P2
    S = np.linalg.inv(P)
    Pm = Pm1 + Pm2
    m = np.dot(S, Pm)
    return m, S


def init_all_gaussian(n_params=2, seed=42, inferenceobj=None, Sfac=1.0, **inf_setup_opts):
    model = Gauss(dim=n_params, seed=seed, noise_cov=0.1 * Sfac)
    prior = dd.Gaussian(m=np.zeros((n_params, )), S=np.eye(n_params) * Sfac, seed=seed+1)
    s = ds.Identity(seed=seed+2)
    g = dg.Default(model=model, prior=prior, summary=s, seed=seed+3)
    obs = np.zeros((1, n_params))  # reseed generator etc. (?)

    res = inferenceobj(g, obs=obs, seed=seed+4, **inf_setup_opts)
    res.reset(seed=seed+4)

    m_true, S_true = simplegaussprod(obs, model.noise_cov, prior.m, prior.S)
    return res, m_true, S_true


def check_gaussian_posterior(posterior, m_true, S_true, atol_mean=0.05, atol_cov=0.05,
                             n_samples=10000):
    if isinstance(posterior, dd.MoG):
        posterior_gauss = posterior.project_to_gaussian()
        posterior_mean = posterior_gauss.m
        posterior_cov = posterior_gauss.S
    elif isinstance(posterior, MAFconditional):
        # Note that if the inference method was seeded, the resulting posteriors
        # will also be seeded upon creation when network.get_density is called.
        posterior_samples = posterior.gen(n_samples)
        posterior_mean = posterior_samples.mean(axis=0)
        posterior_cov = np.cov(posterior_samples.T)
    else:
        raise NotImplementedError

    assert np.allclose(posterior_mean, m_true, atol=atol_mean)
    assert np.allclose(posterior_cov, S_true, atol=atol_cov)


def test_basic_inference(n_params=2, seed=42):
    res, m_true, S_true = init_all_gaussian(seed=seed, n_params=n_params,
                                            inferenceobj=infer.Basic)
    out = res.run(n_train=1000)
    posterior = res.predict(res.obs.reshape(1, -1))
    check_gaussian_posterior(posterior, m_true, S_true)


def test_basic_inference_inputsamples(n_params=2, seed=42, n_pilot=1000):
    model = Gauss(dim=n_params, seed=seed)
    prior = dd.Gaussian(m=np.zeros((n_params,)), S=np.eye(n_params),
                        seed=seed + 1)
    s = ds.Identity(seed=seed + 2)
    g = dg.Default(model=model, prior=prior, summary=s, seed=seed + 3)
    obs = np.zeros((1, n_params))  # reseed generator etc. (?)
    m_true, S_true = simplegaussprod(obs, model.noise_cov, prior.m, prior.S)

    params, stats = g.gen(n_pilot)
    pilot_samples = (params, stats)

    res = infer.Basic(g, obs=obs, seed=seed + 4, pilot_samples=pilot_samples)
    res.reset(seed=seed + 4)

    out = res.run(n_train=1000)
    posterior = res.predict(res.obs.reshape(1, -1))

    check_gaussian_posterior(posterior, m_true, S_true)


def test_snpeB_inference(n_params=2, seed=42):
    res, m_true, S_true = init_all_gaussian(seed=seed, n_params=n_params,
                                            inferenceobj=infer.SNPEB)
    out = res.run(n_train=1000, n_rounds=1)
    posterior = res.predict(res.obs.reshape(1, -1))
    check_gaussian_posterior(posterior, m_true, S_true)


def test_apt_inference_mogprop(n_params=2, seed=47):
    inf_setup_opts = dict(n_components=2)
    res, m_true, S_true = init_all_gaussian(seed=seed, n_params=n_params,
                                            inferenceobj=infer.APT,
                                            **inf_setup_opts)
    out = res.run(n_train=1000, n_rounds=2, proposal='mog',
                  train_on_all=True, silent_fail=False, print_each_epoch=True,
                  reuse_prior_samples=True)
    posterior = res.predict(res.obs.reshape(1, -1))
    check_gaussian_posterior(posterior, m_true, S_true)


def test_apt_inference_gaussprop(n_params=2, seed=47, Sfac=1000.0):
    inf_setup_opts = dict(n_components=2, prior_norm=True)
    res, m_true, S_true = init_all_gaussian(seed=seed, n_params=n_params,
                                            inferenceobj=infer.APT,
                                            Sfac=Sfac,
                                            **inf_setup_opts)
    # 3 rounds to test sample reuse. by default prior samples not reused
    out = res.run(n_train=1500, n_rounds=3, proposal='gaussian',
                  train_on_all=True, silent_fail=False, print_each_epoch=True,
                  reuse_prior_samples=True)

    posterior = res.predict(res.obs.reshape(1, -1))
    check_gaussian_posterior(posterior, m_true, S_true, atol_mean=0.05 * np.sqrt(Sfac), atol_cov=0.05 * Sfac)


def test_apt_inference_atomicprop_mdn(n_params=2, seed=47):
    inf_setup_opts = dict(n_components=2)
    res, m_true, S_true = init_all_gaussian(seed=seed, n_params=n_params,
                                            inferenceobj=infer.APT,
                                            **inf_setup_opts)
    out = res.run(n_train=1020, n_rounds=2, proposal='atomic', n_atoms=10,
                  train_on_all=True, silent_fail=False, print_each_epoch=True, verbose=True, val_frac=0.05)
    posterior = res.predict(res.obs.reshape(1, -1))
    check_gaussian_posterior(posterior, m_true, S_true)


def test_apt_inference_atomicprop_mdn_comb(n_params=2, seed=47):
    inf_setup_opts = dict(n_components=2)
    res, m_true, S_true = init_all_gaussian(seed=seed, n_params=n_params,
                                            inferenceobj=infer.APT,
                                            **inf_setup_opts)
    out = res.run(n_train=1000, n_rounds=2, proposal='atomic_comb',
                  n_atoms=10, train_on_all=True, silent_fail=False, print_each_epoch=True)
    posterior = res.predict(res.obs.reshape(1, -1))
    check_gaussian_posterior(posterior, m_true, S_true)


def test_apt_inference_atomicprop_maf(n_params=2, seed=47):
    inf_setup_opts = dict(density='maf', maf_mode='random', n_mades=5,
                          maf_actfun='tanh', batch_norm=False)
    res, m_true, S_true = init_all_gaussian(seed=seed, n_params=n_params,
                                            inferenceobj=infer.APT,
                                            **inf_setup_opts)
    out = res.run(n_train=1000, n_rounds=2, proposal='atomic', n_atoms=10,
                  train_on_all=True, silent_fail=False, print_each_epoch=True)
    posterior = res.predict(res.obs.reshape(1, -1))
    check_gaussian_posterior(posterior, m_true, S_true)


def test_apt_inference_atomicprop_maf_comb(n_params=2, seed=47):
    inf_setup_opts = dict(density='maf', maf_mode='random', n_mades=5,
                          maf_actfun='tanh', batch_norm=False)
    res, m_true, S_true = init_all_gaussian(seed=seed, n_params=n_params,
                                            inferenceobj=infer.APT,
                                            **inf_setup_opts)
    out = res.run(n_train=1000, n_rounds=2, proposal='atomic_comb', n_atoms=10,
                  train_on_all=True, silent_fail=False, print_each_epoch=True)
    posterior = res.predict(res.obs.reshape(1, -1))
    check_gaussian_posterior(posterior, m_true, S_true)


def test_inference_apt_maf_rnn(n_steps=2, dim_per_t=2, seed=42):
    if theano.config.device == 'cpu':
        return  # need a gpu
    # we're going to reshape a Gaussian observation to be a time series
    # this will test the code but a better test would be a Kalman filter.
    n_params = n_steps * dim_per_t

    # we need to infer a posterior with n_params degrees of freedom, so we need
    # to essentially "record" every data dimension with the RNN
    inf_setup_opts = dict(density='maf', maf_mode='random', n_mades=5,
                          maf_actfun='tanh', batch_norm=False,
                          n_rnn=5 * n_params, input_shape=(n_steps, dim_per_t))
    res, m_true, S_true = init_all_gaussian(seed=seed, n_params=n_params,
                                            inferenceobj=infer.APT,
                                            **inf_setup_opts)
    out = res.run(n_train=5000, n_rounds=2, proposal='atomic', n_atoms=10,
                  train_on_all=True, silent_fail=False, print_each_epoch=True)
    posterior = res.predict(res.obs.reshape(1, -1))
    check_gaussian_posterior(posterior, m_true, S_true)


def test_inference_apt_maf_cnn(rows=2, cols=2, seed=42):
    if theano.config.device == 'cpu':
        return  # need a gpu
    # we're going to reshape a Gaussian observation to be an image
    # this will test the code but a better test would involve correlated x_i.
    # one option would be to try using a very small blob model
    n_params = rows * cols

    # we need to infer a posterior with n_params degrees of freedom, so we need
    # to essentially "record" every data dimension with the CNN
    inf_setup_opts = dict(density='maf', maf_mode='random', n_mades=5,
                          maf_actfun='tanh', batch_norm=False,
                          n_filters=[2 * n_params], filter_sizes=[2],
                          pool_sizes=[1],
                          input_shape=(1, rows, cols))
    res, m_true, S_true = init_all_gaussian(seed=seed, n_params=n_params,
                                            inferenceobj=infer.APT,
                                            **inf_setup_opts)
    out = res.run(n_train=5000, n_rounds=3, proposal='atomic', n_atoms=10,
                  train_on_all=True, silent_fail=False, print_each_epoch=True)
    posterior = res.predict(res.obs.reshape(1, -1))
    check_gaussian_posterior(posterior, m_true, S_true)


def dont_test_apt_inference_atomicprop_maf_normalize(n_params, seed=47):
    # normalization test is not finished yet.
    m = Gauss(dim=n_params, noise_cov=0.1)
    p = dd.Uniform(lower=-0.05 * np.ones(n_params),
                   upper=0.05 * np.ones(n_params))
    s = ds.Identity()
    g = dg.Default(model=m, prior=p, summary=s)
