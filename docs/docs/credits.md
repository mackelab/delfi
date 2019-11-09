# Credits

## Algorithms

`delfi` implements the algorithms from the following papers:

- [Fast ε-free Inference of Simulation Models with Bayesian Conditional Density Estimation (Papamakarios & Murray, 2016)](https://papers.nips.cc/paper/6084-fast-free-inference-of-simulation-models-with-bayesian-conditional-density-estimation), referred to as SNPE-A
- [Flexible statistical inference for mechanistic models of neural dynamics (Lueckmann, Goncalves, Bassetto, Öcal, Nonnenmacher & Macke, 2017)](https://papers.nips.cc/paper/6728-flexible-statistical-inference-for-mechanistic-models-of-neural-dynamics), referred to as SNPE-B
- [Automatic posterior transformation for likelihood free inference (Greenberg, Nonnenmacher & Macke, 2019)](https://arxiv.org/abs/1905.07488), referred to as APT or SNPE-C


##  Code

`delfi.distribution` and `delfi.neuralnet` build on code written by [George Papamakarios](https://github.com/gpapamak/). The separation of classes in `delfi` is partially inspired by the design of [ELFI: Engine for Likelihood-Free Inference](https://github.com/elfi-dev/elfi).

```
{!LICENSE.txt!}
```
