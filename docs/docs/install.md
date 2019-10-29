# Installation

!!! note
    Importantly, this codebase is written for Python 3 and is not backwards compatible.


You can install delfi by cloning from the github repository and using pip:

```
git clone https://github.com/mackelab/delfi.git
cd delfi
pip install -r requirements.txt
pip install -e .
```

Installing delfi as described above will automatically take care of the requirements.

The installation requirements are specified in `setup.py`. Core dependencies are `theano` and `lasagne`. For lasagne, delfi relies on the development version of lasagne (0.2dev) rather than the stable version (0.1) that is available through pip.

To use APT with Gaussian or Mixture-of-Gaussians proposals, you will likely need to make openblas available to theano. You can do this on Debian/Ubuntu using `sudo apt install libopenblas-dev`.
