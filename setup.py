from setuptools import setup

exec(open("delfi/version.py").read())

setup(
    name="delfi",
    version=__version__,
    description="Density estimation likelihood-free inference",
    url="https://github.com/mackelab/delfi",
    author="Mackelab",
    packages=[
        "delfi",
        "delfi.distribution",
        "delfi.distribution.mixture",
        "delfi.generator",
        "delfi.inference",
        "delfi.neuralnet",
        "delfi.neuralnet.layers",
        "delfi.neuralnet.loss",
        "delfi.simulator",
        "delfi.summarystats",
        "delfi.utils",
    ],
    license="BSD",
    install_requires=[
        "dill",
        "lasagne@git+https://github.com/Lasagne/Lasagne#egg=lasagne",
        "numpy",
        "scipy",
        "theano",
        "tqdm",
        "snl@git+https://github.com/mackelab/snl#egg=snl",
    ],
)
