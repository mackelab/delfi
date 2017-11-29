import delfi.distribution as dd
import delfi.generator as dg
import delfi.summarystats as ds
import numpy as np
import theano
import theano.tensor as tt

from delfi.simulator.Gauss import Gauss
from delfi.neuralnet.NeuralNet import NeuralNet
from delfi.neuralnet.Trainer import Trainer

dtype = theano.config.floatX


def test_trainer_updates():
    n_components = 1
    n_params = 2
    seed = 42
    svi = True

    m = Gauss(dim=n_params)
    p = dd.Gaussian(m=np.zeros((n_params, )), S=np.eye(n_params))
    s = ds.Identity()
    g = dg.Default(model=m, prior=p, summary=s)

    nn = NeuralNet(
        n_components=n_components,
        n_hiddens=[10],
        n_inputs=n_params,
        n_outputs=n_params,
        seed=seed,
        svi=svi)
    loss = -tt.mean(nn.lprobs)

    trn_inputs = [nn.params, nn.stats]
    trn_data = g.gen(100)  # params, stats
    trn_data = tuple(x.astype(dtype) for x in trn_data)

    t = Trainer(network=nn, loss=loss, trn_data=trn_data, trn_inputs=trn_inputs)

    # single update
    outputs = t.make_update(*trn_data)

    # training
    outputs = t.train(100, 50)
