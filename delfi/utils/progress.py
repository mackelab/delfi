from tqdm import tqdm, tqdm_notebook


class no_tqdm(object):
    def __enter__(self):
        class blank(object):
            def update(self, x):
                pass
        return blank()

    def __exit__(self, type, value, traceback):
        pass

    def update(self, i):
        pass


def progressbar(*args, **kwargs):
    """Creates a tqdm instance for a notebook or command line

    There is an open issue to support this as part of tqdm, see:
    https://github.com/tqdm/tqdm/issues/234
    https://github.com/tqdm/tqdm/issues/372
    """
    try:
        from IPython import get_ipython
        from ipywidgets import FloatProgress
        ipython = get_ipython()
        if not ipython or ipython.__class__.__name__ != 'ZMQInteractiveShell':
            raise RuntimeError
        return tqdm_notebook(*args, **kwargs)
    except BaseException:
        return tqdm(*args, **kwargs)
