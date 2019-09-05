import numpy as np
import theano
import lasagne

from delfi.neuralnet.NeuralNet import NeuralNet

dtype = theano.config.floatX


def test_lprobs():
    n_components = 2
    seed = 42
    svi = False

    nn = NeuralNet(n_components=n_components, n_hiddens=[10], n_inputs=1,
                   n_outputs=1, seed=seed, svi=svi)

    eval_lprobs = theano.function([nn.params, nn.stats], nn.lprobs)

    res = eval_lprobs(np.array([[1.], [2.]], dtype=dtype),
                      np.array([[1.], [2.]], dtype=dtype))

    mog = nn.get_mog(np.array([[1.]], dtype=dtype))


def test_diag_precision_bounds():
    n_components = 2
    seed = 42
    svi = False
    min_precisions = np.array([0.1, 2.0, 15.0])

    nn = NeuralNet(n_components=n_components, n_hiddens=[10], n_inputs=3,
                   n_outputs=3, seed=seed, svi=svi, min_precisions=None)

    nn_bounded = NeuralNet(n_components=n_components, n_hiddens=[10],
                           n_inputs=3, n_outputs=3, seed=seed, svi=svi,
                           min_precisions=min_precisions)
    mog_bounded = nn_bounded.get_mog(np.array(np.ones((1, 3)), dtype=dtype))
    for x in mog_bounded.xs:
        assert np.allclose(0.0, np.maximum(0.0, min_precisions - np.diag(x.P)))


def test_conv():
    """Test multichannel convolution with bypass"""
    n_components = 1
    seed = 42
    L = 20
    channels = 3
    bypass = 1
    svi = False

    n_inputs = channels * L ** 2 + bypass

    nn = NeuralNet(n_components=n_components, n_filters=[8, 8], n_hiddens=[10],
                   n_inputs=n_inputs, input_shape=(channels, L, L),
                   n_bypass=1, n_outputs=1,
                   seed=seed, svi=svi)

    eval_lprobs = theano.function([nn.params, nn.stats], nn.lprobs)

    conv_sizes = [lasagne.layers.get_output_shape(
        nn.layer['conv_'+str(i+1)]) for i in range(2)]

    res = eval_lprobs(np.array([[1.], [2.]], dtype=dtype),
                      np.random.normal(size=(2, n_inputs)).astype(dtype))

    mog = nn.get_mog(np.random.normal(size=(1, n_inputs)).astype(dtype))

