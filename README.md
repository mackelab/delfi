# delfi

[![Build Status](https://travis-ci.org/mackelab/delfi.svg?branch=master)](https://travis-ci.org/mackelab/delfi) [![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://www.mackelab.org/delfi/)


delfi is a Python package for density estimation likelihood-free inference. In particular it includes several algorithms for Sequential Neural Posterior Estimation (SNPE).

Several inference algorithms are implemented:
* A basic version of a likelihood-free inference algorithm that uses a mixture-density network to approximate the posterior density
* The SNPE-A algorithm proposed in the paper [Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation (Papamakarios & Murray, 2016)](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation)
* SNPE-B, as proposed in the paper [Flexible statistical inference for mechanistic models of neural dynamics (Lueckmann, Goncalves, Bassetto, Öcal, Nonnenmacher & Macke, 2017)](https://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics)
* APT as proposed in [Automatic posterior transformation for likelihood free inference](https://icml.cc/Conferences/2019/ScheduleMultitrack?event=4268)


## Documentation

Please note that the code in this repository is still experimental. An early-stage documentation including [installation instructions](http://www.mackelab.org/delfi/installation.html) and a [guide on how to get started](http://www.mackelab.org/delfi/notebooks/quickstart.html) is available at [http://www.mackelab.org/delfi/](http://www.mackelab.org/delfi/).
