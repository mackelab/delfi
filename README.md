# delfi

[![Build Status](https://travis-ci.org/mackelab/delfi.svg?branch=master)](https://travis-ci.org/mackelab/delfi) [![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](http://www.mackelab.org/delfi/)


delfi is a Python package for **d**ensity **e**stimation **l**ikelihood-free **i**nference.
 
Includes several algorithms for **s**equential **n**eural **p**osterior **e**stimation (SNPE):
* A basic version of a likelihood-free inference algorithm that uses a mixture-density network to approximate the posterior density
* The SNPE-A algorithm proposed in the paper [Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation (Papamakarios & Murray, 2016)](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation)
* SNPE-B, as proposed in the paper [Flexible statistical inference for mechanistic models of neural dynamics (Lueckmann, Goncalves, Bassetto, Öcal, Nonnenmacher & Macke, 2017)](https://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics)
* APT as proposed in [Automatic posterior transformation for likelihood free inference (Greenberg, Nonnenmacher & Macke, 2019)](https://arxiv.org/abs/1905.07488)

## Documentation

Please note that the code in this repository is still experimental. An early-stage documentation including [installation instructions](http://www.mackelab.org/delfi/installation.html) and a [guide on how to get started](http://www.mackelab.org/delfi/notebooks/quickstart.html) is available at [http://www.mackelab.org/delfi/](http://www.mackelab.org/delfi/).

## Alternative approaches

As an alternative to directly estimating the posterior on parameters given data, it is also possible to estimate the likelihood of data given parameters, and then subsequently draw posterior samples using MCMC. [Sequential Neural Likelihood (Papamakarios, Sterratt & Murray, 2018)](https://arxiv.org/abs/1805.07226) is a powerful technique employing this strategy. Depending on the problem, SNL can be more or less effective than SNPE techniques such as APT. Code for SNL is available from the [original repository](https://github.com/gpapamak/snl) or as a [python 3 package](https://github.com/mnonnenm/SNL_py3port/tree/master).  