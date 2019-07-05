import numpy as np

from delfi.simulator import Gauss, TwoMoons, MoGDistractors, TransformedSimulator


def test_distractors():
    dim = 2
    s = MoGDistractors(dim=dim)
    n_samples = 10
    thetas = np.random.rand(n_samples, dim)
    sample_list = s.gen(thetas)
    assert len(sample_list) == n_samples
    assert isinstance(
        sample_list[0][0], dict), 'the entries should be dictionaries'


def test_twomoons():
    s = TwoMoons()

    n_samples = 10
    thetas = np.random.rand(n_samples, 2)
    sample_list = s.gen(thetas)

    assert len(sample_list) == n_samples
    assert isinstance(
        sample_list[0][0], dict), 'the entries should be dictionaries'


def test_gauss_1d_simulator_output():
    """Test the output of the simulator using the example of a 1D Gaussian
    """
    dim = 1
    s = Gauss(dim=dim)

    n_samples = 10
    thetas = np.tile(np.array([0.]), (n_samples, 1))
    sample_list = s.gen(thetas)

    assert len(
        sample_list) == n_samples, 'the list should have as many entries as there are samples'
    assert isinstance(
        sample_list[0][0], dict), 'the entries should be dictionaries'


def test_gauss_2d_data_dimension():
    """Test the data dimensionality output of the Gauss Simulator using a 2D Gaussian
    """
    dim = 2
    s = Gauss(dim=dim)

    n_samples = 10
    thetas = np.tile(np.array([0., 1.]), (n_samples, 1))
    sample_list = s.gen(thetas)

    assert sample_list[0][0]['data'].shape == (dim, ), \
        'the dimensionality of the data is wrong. ' \
        'should be {} is {}'.format((dim, 1), sample_list[0][0]['data'].shape)


def test_TransformedSimulator(seed=5, nsamples=1000, dim=2):
    s = Gauss(dim=dim)

    bijection = lambda x: np.log(x)
    inverse_bijection = lambda x: np.exp(x)

    theta = np.random.rand(nsamples, dim)
    theta_transformed = bijection(theta)

    s.reseed(seed)
    x = np.array([z[0]['data'] for z  in s.gen(theta)])

    s_transformed = TransformedSimulator(s, inverse_bijection)
    s_transformed.reseed(seed)
    x_from_transformed_theta = np.array([z[0]['data'] for z in s_transformed.gen(theta_transformed)])

    assert np.allclose(x, x_from_transformed_theta, atol=1e-8), "data don't match after parameter transformation"