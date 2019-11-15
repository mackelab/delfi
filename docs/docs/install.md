# Installation

You can install `delfi` by cloning from the [GitHub repository](https://github.com/mackelab/delfi) and using pip:

```
git clone https://github.com/mackelab/delfi.git
cd delfi
pip install -r requirements.txt
pip install -e .
```

Installing `delfi` as described above will automatically take care of the requirements. In total, this installation will typically take less than a minute.


## Small print

`delfi` is written for Python 3 and not compatible with older versions.

Core dependencies are the packages `theano` and `lasagne`. For `lasagne`, `delfi` relies on the development version of `lasagne` (0.2dev) rather than the stable version (0.1) that is available through pip.

To use the APT inference algorithm with Gaussian or Mixture-of-Gaussians proposals, you will likely need to make openblas available for `theano`. If openblas is missing on your system and you happen to use Debian/Ubuntu you can install it with `sudo apt install libopenblas-dev`.
