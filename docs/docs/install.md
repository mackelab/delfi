# Installation

## Quick start
You can install `delfi` by cloning from the [GitHub repository](https://github.com/mackelab/delfi) and using pip:

```
git clone https://github.com/mackelab/delfi.git
cd delfi
pip install -r requirements.txt
pip install -e .
```

Installing `delfi` as described above will automatically take care of the requirements. In total, this installation will typically take less than a minute.

## GPU support from a fresh Anaconda installation
To install delfi and all prerequisites totally from scratch, follow these steps for Ubuntu 19.10 (last updated Jan 20, 2020).

* Install the correct CUDA driver for your GPU, if not already installed
* Install ```libopenblas-dev``` if you haven't done so already (see below)
* Install CUDA and cuDNN if not already installed
* Download, install and activate Anaconda for python 3 ([instructions](https://docs.anaconda.com/anaconda/install/linux/))
* Close any terminal windows and make a new one, and run these commands:
```
conda install ipython numpy scipy pytest matplotlib mkl mkl-service
pip install theano
conda install pygpu
echo -e "\n[blas]\nldflags = -lopenblas\n" >> ~/.theanorc
```
* Open a terminal, change to the base directory of the delfi repository, and run these commands:
```
pip install -r requirements.txt
pip install -e .
```

## Small print

`delfi` is written for Python 3 and not compatible with older versions.

Core dependencies are the packages `theano` and `lasagne`. For `lasagne`, `delfi` relies on the development version of `lasagne` (0.2dev) rather than the stable version (0.1) that is available through pip.

To use the APT inference algorithm with Gaussian or Mixture-of-Gaussians proposals, you will likely need to make openblas available for `theano`. If openblas is missing on your system and you happen to use Debian/Ubuntu you can install it with `sudo apt install libopenblas-dev`.
