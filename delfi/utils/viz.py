import collections
import delfi.utils.colormaps as cmaps
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import six
import time

from scipy.stats import gaussian_kde

try:
    collectionsAbc = collections.abc
except:
    collectionsAbc = collections

try:
    from seaborn.utils import despine
except:
    def despine(*args, **kwargs):
        pass


def loss(losses, key='trn', loss_clipping=1000., title=''):
    """Given an info dict, plot loss"""

    x = np.array(losses[key + '_iter'])
    y = np.array(losses[key + '_val'])

    clip_idx = np.where(y > loss_clipping)[0]
    if len(clip_idx) > 0:
        print(
            'warning: loss exceeds threshold of {:.2f} in total {} time(s); values will be clipped'.format(
                loss_clipping,
                len(clip_idx)))

    y[clip_idx] = loss_clipping

    options = {}
    options['title'] = title
    options['xlabel'] = r'iteration'
    options['ylabel'] = r'loss'

    fig, ax = plt.subplots(1, 1)
    ax.semilogx(x, y, 'b')
    ax.set_xlabel(options['xlabel'])
    ax.set_ylabel(options['ylabel'])

    return fig, ax


def dist(dist, title=''):
    """Given dist, plot histogram"""
    options = {}
    options['title'] = title
    options['xlabel'] = r'bin'
    options['ylabel'] = r'distance'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_samples = len(dist)
    ax.hist(dist, bins=int(np.sqrt(n_samples)))
    ax.set_xlabel(options['xlabel'])
    ax.set_ylabel(options['ylabel'])
    ax.set_title(options['title'])
    return fig, ax


def info(info, html=False, title=None):
    """Given info dict, produce info text"""
    if title is None:
        infotext = u''
    else:
        if html:
            infotext = u'<b>{}</b><br>'.format(title)
        else:
            infotext = u'{}\n'.format(title)

    for key, value in info.items():
        if key not in ['losses']:
            infotext += u'{} : {}'.format(key, value)
            if html:
                infotext += '<br>'
            else:
                infotext += '\n'

    return infotext


def probs2contours(probs, levels):
    """Takes an array of probabilities and produces an array of contours at specified percentile levels

    Parameters
    ----------
    probs : array
        Probability array. doesn't have to sum to 1, but it is assumed it contains all the mass
    levels : list
        Percentile levels, have to be in [0.0, 1.0]

    Return
    ------
    Array of same shape as probs with percentile labels
    """
    # make sure all contour levels are in [0.0, 1.0]
    levels = np.asarray(levels)
    assert np.all(levels <= 1.0) and np.all(levels >= 0.0)

    # flatten probability array
    shape = probs.shape
    probs = probs.flatten()

    # sort probabilities in descending order
    idx_sort = probs.argsort()[::-1]
    idx_unsort = idx_sort.argsort()
    probs = probs[idx_sort]

    # cumulative probabilities
    cum_probs = probs.cumsum()
    cum_probs /= cum_probs[-1]

    # create contours at levels
    contours = np.ones_like(cum_probs)
    levels = np.sort(levels)[::-1]
    for level in levels:
        contours[cum_probs <= level] = level

    # make sure contours have the order and the shape of the original
    # probability array
    contours = np.reshape(contours[idx_unsort], shape)

    return contours


def plot_pdf(pdf1, lims=None, pdf2=None, gt=None, contours=False, levels=(0.68, 0.95),
             resolution=500, labels_params=None, ticks=False, diag_only=False,
             diag_only_cols=4, diag_only_rows=4, figsize=(5, 5), fontscale=1,
             partial=False, samples=None, col1='k', col2='b', col3='g'):
    """Plots marginals of a pdf, for each variable and pair of variables.

    Parameters
    ----------
    pdf1 : object
    lims : array
    pdf2 : object (or None)
        If not none, visualizes pairwise marginals for second pdf on lower diagonal
    contours : bool
    levels : tuple
        For contours
    resolution
    labels_params : array of strings
    ticks: bool
        If True, includes ticks in plots
    diag_only : bool
    diag_only_cols : int
        Number of grid columns if only the diagonal is plotted
    diag_only_rows : int
        Number of grid rows if only the diagonal is plotted
    fontscale: int
    partial: bool
        If True, plots partial posterior with at the most 3 parameters.
        Only available if `diag_only` is False
    samples: array
        If given, samples of a distribution are plotted along `pdf`.
        If given, `pdf` is plotted with default `levels` (0.68, 0.95), if provided `levels` is None.
        If given, `lims` is overwritten and taken to be the respective
        limits of the samples in each dimension.
    col1 : str
        color 1
    col2 : str
        color 2
    col3 : str
        color 3 (for pdf2 if provided)
    """

    pdfs = (pdf1, pdf2)
    colrs = (col2, col3)

    if not (pdf1 is None or pdf2 is None):
        assert pdf1.ndim == pdf2.ndim

    if samples is not None:
        contours = True
        if levels is None:
            levels = (0.68, 0.95)

    if samples is not None and lims is None:
        lims_min = np.min(samples, axis=1)
        lims_max = np.max(samples, axis=1)
        lims = np.asarray(lims)
        lims = np.concatenate(
            (lims_min.reshape(-1, 1), lims_max.reshape(-1, 1)), axis=1)
    else:
        lims = np.asarray(lims)
        lims = np.tile(lims, [pdf1.ndim, 1]) if lims.ndim == 1 else lims

    if pdf1.ndim == 1:

        fig, ax = plt.subplots(1, 1, facecolor='white', figsize=figsize)

        if samples is not None:
            ax.hist(samples[i, :], bins=100, density=True,
                    color=col1,
                    edgecolor=col1)

        xx = np.linspace(lims[0, 0], lims[0, 1], resolution)

        for pdf, col in zip(pdfs, colrs):
            if pdf is not None:
                pp = pdf.eval(xx[:, np.newaxis], log=False)
                ax.plot(xx, pp, color=col)
        ax.set_xlim(lims[0])
        ax.set_ylim([0, ax.get_ylim()[1]])
        if gt is not None:
            ax.vlines(gt, 0, ax.get_ylim()[1], color='r')

        if ticks:
            ax.get_yaxis().set_tick_params(which='both', direction='out')
            ax.get_xaxis().set_tick_params(which='both', direction='out')
            ax.set_xticks(np.linspace(lims[0, 0], lims[0, 1], 2))
            ax.set_yticks(np.linspace(min(pp), max(pp), 2))
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        else:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    else:

        if not diag_only:
            if partial:
                rows = min(3, pdf1.ndim)
                cols = min(3, pdf1.ndim)
            else:
                rows = pdf1.ndim
                cols = pdf1.ndim
        else:
            cols = diag_only_cols
            rows = diag_only_rows
            r = 0
            c = -1

        fig, ax = plt.subplots(rows, cols, facecolor='white', figsize=figsize)
        ax = ax.reshape(rows, cols)

        for i in range(rows):
            for j in range(cols):

                if i == j:
                    if samples is not None:
                        ax[i, j].hist(samples[i, :], bins=100, density=True,
                                      color=col1,
                                      edgecolor=col1)
                    xx = np.linspace(lims[i, 0], lims[i, 1], resolution)

                    for pdf, col in zip(pdfs, colrs):
                        if pdf is not None:
                            pp = pdf.eval(xx, ii=[i], log=False)

                            if diag_only:
                                c += 1
                            else:
                                r = i
                                c = j

                    for pdf, col in zip(pdfs, colrs):
                        if pdf is not None:
                            pp = pdf.eval(xx, ii=[i], log=False)
                            ax[r, c].plot(xx, pp, color=col)

                    ax[r, c].set_xlim(lims[i])
                    ax[r, c].set_ylim([0, ax[r, c].get_ylim()[1]])

                    if gt is not None:
                        ax[r, c].vlines(
                            gt[i], 0, ax[r, c].get_ylim()[1], color='r')

                    if ticks:
                        ax[r, c].get_yaxis().set_tick_params(
                            which='both', direction='out', labelsize=fontscale * 15)
                        ax[r, c].get_xaxis().set_tick_params(
                            which='both', direction='out', labelsize=fontscale * 15)
#                         ax[r, c].locator_params(nbins=3)
                        ax[r, c].set_xticks(np.linspace(
                            lims[i, 0]+0.15*np.abs(lims[i, 0]-lims[j, 1]), lims[j, 1]-0.15*np.abs(lims[i, 0]-lims[j, 1]), 2))
                        ax[r, c].set_yticks(np.linspace(0+0.15*np.abs(0-max(pp)), max(pp)-0.15*np.abs(0-max(pp)), 2))
                        ax[r, c].xaxis.set_major_formatter(
                            mpl.ticker.FormatStrFormatter('%.1f'))
                        ax[r, c].yaxis.set_major_formatter(
                            mpl.ticker.FormatStrFormatter('%.1f'))
                    else:
                        ax[r, c].get_xaxis().set_ticks([])
                        ax[r, c].get_yaxis().set_ticks([])

                    if labels_params is not None:
                        ax[r, c].set_xlabel(
                            labels_params[i], fontsize=fontscale * 20)
                    else:
                        ax[r, c].set_xlabel([])

                    x0, x1 = ax[r, c].get_xlim()
                    y0, y1 = ax[r, c].get_ylim()
                    ax[r, c].set_aspect((x1 - x0) / (y1 - y0))

                    if partial and i == rows - 1:
                        ax[i, j].text(x1 + (x1 - x0) / 6., (y0 + y1) /
                                      2., '...', fontsize=fontscale * 25)
                        plt.text(x1 + (x1 - x0) / 8.4, y0 - (y1 - y0) /
                                 6., '...', fontsize=fontscale * 25, rotation=-45)

                else:
                    if diag_only:
                        continue

                    if i < j:
                        pdf = pdfs[0]
                    else:
                        pdf = pdfs[1]

                    if pdf is None:
                        ax[i, j].get_yaxis().set_visible(False)
                        ax[i, j].get_xaxis().set_visible(False)
                        ax[i, j].set_axis_off()
                        continue

                    if samples is not None:
                        H, xedges, yedges = np.histogram2d(
                            samples[i, :], samples[j, :], bins=30, range=[
                            [lims[i, 0], lims[i, 1]], [lims[j, 0], lims[j, 1]]], density=True)
                        ax[i, j].imshow(H, origin='lower', extent=[
                                        yedges[0], yedges[-1], xedges[0], xedges[-1]])

                    xx = np.linspace(lims[i, 0], lims[i, 1], resolution)
                    yy = np.linspace(lims[j, 0], lims[j, 1], resolution)
                    X, Y = np.meshgrid(xx, yy)
                    xy = np.concatenate(
                        [X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=1)
                    pp = pdf.eval(xy, ii=[i, j], log=False)
                    pp = pp.reshape(list(X.shape))
                    if contours:
                        ax[i, j].contour(Y, X, probs2contours(
                            pp, levels), levels, colors=('w', 'y'))
                    else:
                        ax[i, j].imshow(pp.T, origin='lower',
                                        extent=[lims[j, 0], lims[j, 1], lims[i, 0], lims[i, 1]],
                                        aspect='auto', interpolation='none')
                    ax[i, j].set_xlim(lims[j])
                    ax[i, j].set_ylim(lims[i])

                    if gt is not None:
                        ax[i, j].plot(gt[j], gt[i], 'r.', ms=10,
                                      markeredgewidth=0.0)

                    ax[i, j].get_xaxis().set_ticks([])
                    ax[i, j].get_yaxis().set_ticks([])
                    ax[i, j].set_axis_off()

                    x0, x1 = ax[i, j].get_xlim()
                    y0, y1 = ax[i, j].get_ylim()
                    ax[i, j].set_aspect((x1 - x0) / (y1 - y0))

                    if partial and j == cols - 1:
                        ax[i, j].text(x1 + (x1 - x0) / 6., (y0 + y1) /
                                      2., '...', fontsize=fontscale * 25)

                if diag_only and c == cols - 1:
                    c = -1
                    r += 1

    return fig, ax


def plot_hist_marginals(data, lims=None, gt=None):
    """Plots marginal histograms and pairwise scatter plots of a dataset"""
    n_bins = int(np.sqrt(data.shape[0]))

    if data.ndim == 1:
        fig, ax = plt.subplots(1, 1, facecolor='white')
        ax.hist(data, n_bins, density=True)
        ax.set_ylim([0, ax.get_ylim()[1]])
        if lims is not None:
            ax.set_xlim(lims)
        if gt is not None:
            ax.vlines(gt, 0, ax.get_ylim()[1], color='r')

    else:
        n_dim = data.shape[1]
        fig, ax = plt.subplots(n_dim, n_dim, facecolor='white')
        ax = np.array([[ax]]) if n_dim == 1 else ax

        if lims is not None:
            lims = np.asarray(lims)
            lims = np.tile(lims, [n_dim, 1]) if lims.ndim == 1 else lims

        for i in range(n_dim):
            for j in range(n_dim):

                if i == j:
                    ax[i, j].hist(data[:, i], n_bins, density=True)
                    ax[i, j].set_ylim([0, ax[i, j].get_ylim()[1]])
                    if lims is not None:
                        ax[i, j].set_xlim(lims[i])
                    if gt is not None:
                        ax[i, j].vlines(
                            gt[i], 0, ax[i, j].get_ylim()[1], color='r')

                else:
                    ax[i, j].plot(data[:, i], data[:, j], 'k.', ms=2)
                    if lims is not None:
                        ax[i, j].set_xlim(lims[i])
                        ax[i, j].set_ylim(lims[j])
                    if gt is not None:
                        ax[i, j].plot(gt[i], gt[j], 'r.', ms=8)

    return fig, ax



def _update(d, u):
    # https://stackoverflow.com/a/3233356
    for k, v in six.iteritems(u):
        dv = d.get(k, {})
        if not isinstance(dv, collectionsAbc.Mapping):
            d[k] = v
        elif isinstance(v, collectionsAbc.Mapping):
            d[k] = _update(dv, v)
        else:
            d[k] = v
    return d


def _format_axis(ax, xhide=True, yhide=True, xlabel='', ylabel='',
        tickformatter=None):
    for loc in ['right', 'top', 'left', 'bottom']:
        ax.spines[loc].set_visible(False)
    if xhide:
        ax.set_xlabel('')
        ax.xaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(labelbottom=False)
    if yhide:
        ax.set_ylabel('')
        ax.yaxis.set_ticks_position('none')
        ax.yaxis.set_tick_params(labelleft=False)
    if not xhide:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_tick_params(labelbottom=True)
        if tickformatter is not None:
            ax.xaxis.set_major_formatter(tickformatter)
        ax.spines['bottom'].set_visible(True)
    if not yhide:
        ax.set_ylabel(ylabel)
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_tick_params(labelleft=True)
        if tickformatter is not None:
            ax.yaxis.set_major_formatter(tickformatter)
        ax.spines['left'].set_visible(True)
    return ax


def samples_nd(samples, points=[], **kwargs):
    """Plot samples and points

    See `opts` below for available keyword arguments.
    """
    opts = {
        # what to plot on triagonal and diagonal subplots
        'upper': 'hist',   # hist/scatter/None
        'diag': 'hist',    # hist/None
        #'lower': None,     # hist/scatter/None  # TODO: implement

        # title and legend
        'title': None,
        'legend': False,

        # labels
        'labels': [],         # for dimensions
        'labels_points': [],  # for points
        'labels_samples': [], # for samples

        # colors
        'samples_colors': plt.rcParams['axes.prop_cycle'].by_key()['color'],
        'points_colors': plt.rcParams['axes.prop_cycle'].by_key()['color'],

        # subset
        'subset': None,

        # axes limits
        'limits': [],

        # ticks
        'ticks': [],
        'tickformatter': mpl.ticker.FormatStrFormatter('%g'),
        'tick_labels': None,

        # options for hist
        'hist_diag': {
            'alpha': 1.,
            'bins': 25,
            'density': False,
            'histtype': 'step'
        },
        'hist_offdiag': {
            #'edgecolor': 'none',
            #'linewidth': 0.0,
            'bins': 25,
        },

        # options for kde
        'kde_diag': {
            'bw_method': 'scott',
            'bins': 100,
            'color': 'black'
        },
        'kde_offdiag': {
            'bw_method': 'scott',
            'bins': 25
        },

        # options for contour
        'contour_offdiag': {
            'levels': [0.68]
        },

        # options for scatter
        'scatter_offdiag': {
            'alpha': 0.5,
            'edgecolor': 'none',
            'rasterized': False,
        },

        # options for plot
        'plot_offdiag': {},

        # formatting points (scale, markers)
        'points_diag': {
        },
        'points_offdiag': {
            'marker': '.',
            'markersize': 20,
        },

        # matplotlib style
        'style': os.path.join(os.path.dirname(__file__), 'matplotlibrc'),

        # other options
        'fig_size': (10, 10),
        'fig_bg_colors':
            {'upper': None,
             'diag': None,
             'lower': None},
        'fig_subplots_adjust': {
            'top': 0.9,
        },
        'subplots': {
        },
        'despine': {
            'offset': 5,
        },
        'title_format': {
            'fontsize': 16
        },
    }
    # TODO: add color map support
    # TODO: automatically determine good bin sizes for histograms
    # TODO: get rid of seaborn dependency for despine
    # TODO: add legend (if legend is True)

    samples_nd.defaults = opts.copy()
    opts = _update(opts, kwargs)

    # Prepare samples
    if type(samples) != list:
        samples = [samples]

    # Prepare points
    if type(points) != list:
        points = [points]
    points = [np.atleast_2d(p) for p in points]

    # Dimensions
    dim = samples[0].shape[1]
    num_samples = samples[0].shape[0]

    # TODO: add asserts checking compatiblity of dimensions

    # Prepare labels
    if opts['labels'] == [] or opts['labels'] is None:
        labels_dim = ['dim {}'.format(i+1) for i in range(dim)]
    else:
        labels_dim = opts['labels']

    # Prepare limits
    if opts['limits'] == [] or opts['limits'] is None:
        limits = []
        for d in range(dim):
            min = +np.inf
            max = -np.inf
            for sample in samples:
                min_ = sample[:, d].min()
                min = min_ if min_ < min else min
                max_ = sample[:, d].max()
                max = max_ if max_ > max else max
            limits.append([min, max])
    else:
        if len(opts['limits']) == 1:
            limits = [opts['limits'][0] for _ in range(dim)]
        else:
            limits = opts['limits']

    # Prepare ticks
    if opts['ticks'] == [] or opts['ticks'] is None:
        ticks = None
    else:
        if len(opts['ticks']) == 1:
            ticks = [opts['ticks'][0] for _ in range(dim)]
        else:
            ticks = opts['ticks']

    # Prepare diag/upper/lower
    if type(opts['diag']) is not list:
        opts['diag'] = [opts['diag'] for _ in range(len(samples))]
    if type(opts['upper']) is not list:
        opts['upper'] = [opts['upper'] for _ in range(len(samples))]
    #if type(opts['lower']) is not list:
    #    opts['lower'] = [opts['lower'] for _ in range(len(samples))]
    opts['lower'] = None

    # Style
    if opts['style'] in ['dark', 'light']:
        style = os.path.join(
            os.path.dirname(__file__),
            'matplotlib_{}.style'.format(opts['style']))
    else:
        style = opts['style'];

    # Apply custom style as context
    with mpl.rc_context(fname=style):

        # Figure out if we subset the plot
        subset = opts['subset']
        if subset is None:
            rows = cols = dim
            subset = [i for i in range(dim)]
        else:
            if type(subset) == int:
                subset = [subset]
            elif type(subset) == list:
                pass
            else:
                raise NotImplementedError
            rows = cols = len(subset)

        fig, axes = plt.subplots(rows, cols, figsize=opts['fig_size'], **opts['subplots'])
        axes = axes.reshape(rows, cols)

        # Style figure
        fig.subplots_adjust(**opts['fig_subplots_adjust'])
        fig.suptitle(opts['title'], **opts['title_format'])

        # Style axes
        row_idx = -1
        for row in range(dim):
            if row not in subset:
                continue
            else:
                row_idx += 1

            col_idx = -1
            for col in range(dim):
                if col not in subset:
                    continue
                else:
                    col_idx += 1

                if row == col:
                    current = 'diag'
                elif row < col:
                    current = 'upper'
                else:
                    current = 'lower'

                ax = axes[row_idx, col_idx]
                plt.sca(ax)

                # Background color
                if current in opts['fig_bg_colors'] and \
                    opts['fig_bg_colors'][current] is not None:
                    ax.set_facecolor(
                        opts['fig_bg_colors'][current])

                # Axes
                if opts[current] is None:
                    ax.axis('off')
                    continue

                # Limits
                if limits is not None:
                    ax.set_xlim(
                        (limits[col][0], limits[col][1]))
                    if current is not 'diag':
                        ax.set_ylim(
                            (limits[row][0], limits[row][1]))
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()

                # Ticks
                if ticks is not None:
                    ax.set_xticks(
                        (ticks[col][0], ticks[col][1]))
                    if current is not 'diag':
                        ax.set_yticks(
                            (ticks[row][0], ticks[row][1]))

                # Despine
                despine(ax=ax, **opts['despine'])

                # Formatting axes
                if current == 'diag':  # off-diagnoals
                    if opts['lower'] is None or col == dim-1:
                        _format_axis(ax, xhide=False, xlabel=labels_dim[col],
                            yhide=True, tickformatter=opts['tickformatter'])
                    else:
                        _format_axis(ax, xhide=True, yhide=True)
                else:  # off-diagnoals
                    if row == dim-1:
                        _format_axis(ax, xhide=False, xlabel=labels_dim[col],
                            yhide=True, tickformatter=opts['tickformatter'])
                    else:
                        _format_axis(ax, xhide=True, yhide=True)
                if opts['tick_labels'] is not None:
                    ax.set_xticklabels(
                        (str(opts['tick_labels'][col][0]), str(opts['tick_labels'][col][1])))

                # Diagonals
                if current == 'diag':
                    if len(samples) > 0:
                        for n, v in enumerate(samples):
                            if opts['diag'][n] == 'hist':
                                h = plt.hist(
                                    v[:, row],
                                    color=opts['samples_colors'][n],
                                    **opts['hist_diag']
                                )
                            elif opts['diag'][n] == 'kde':
                                density = gaussian_kde(
                                    v[:, row],
                                    bw_method=opts['kde_diag']['bw_method'])
                                xs = np.linspace(xmin, xmax, opts['kde_diag']['bins'])
                                ys = density(xs)
                                h = plt.plot(xs, ys,
                                    color=opts['samples_colors'][n],
                                )
                            else:
                                pass

                    if len(points) > 0:
                        extent = ax.get_ylim()
                        for n, v in enumerate(points):
                            h = plt.plot(
                                [v[:, row], v[:, row]],
                                extent,
                                color=opts['points_colors'][n],
                                **opts['points_diag']
                            )

                # Off-diagonals
                else:

                    if len(samples) > 0:
                        for n, v in enumerate(samples):
                            if opts['upper'][n] == 'hist' or opts['upper'][n] == 'hist2d':
                                hist, xedges, yedges = np.histogram2d(
                                    v[:, col], v[:, row], range=[
                                        [limits[col][0], limits[col][1]],
                                        [limits[row][0], limits[row][1]]],
                                    **opts['hist_offdiag'])
                                h = plt.imshow(hist.T,
                                    origin='lower',
                                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                                    aspect='auto'
                                )

                            elif opts['upper'][n] in ['kde', 'kde2d', 'contour', 'contourf']:
                                density = gaussian_kde(v[:, [col, row]].T, bw_method=opts['kde_offdiag']['bw_method'])
                                X, Y = np.meshgrid(np.linspace(limits[col][0], limits[col][1], opts['kde_offdiag']['bins']),
                                                   np.linspace(limits[row][0], limits[row][1], opts['kde_offdiag']['bins']))
                                positions = np.vstack([X.ravel(), Y.ravel()])
                                Z = np.reshape(density(positions).T, X.shape)

                                if opts['upper'][n] == 'kde' or opts['upper'][n] == 'kde2d':
                                    h = plt.imshow(Z,
                                        extent=[limits[col][0], limits[col][1], limits[row][0], limits[row][1]],
                                        origin='lower',
                                        aspect='auto',
                                    )
                                elif opts['upper'][n] == 'contour':
                                    Z = (Z - Z.min())/(Z.max() - Z.min())
                                    h = plt.contour(X, Y, Z,
                                        origin='lower',
                                        extent=[limits[col][0], limits[col][1], limits[row][0], limits[row][1]],
                                        colors=opts['samples_colors'][n],
                                        **opts['contour_offdiag']
                                    )
                                else:
                                    pass
                            elif opts['upper'][n] == 'scatter':
                                h = plt.scatter(
                                    v[:, col], v[:, row],
                                    color=opts['samples_colors'][n],
                                    **opts['scatter_offdiag']
                                )
                            elif opts['upper'][n] == 'plot':
                                h = plt.plot(
                                    v[:, col], v[:, row],
                                    color=opts['samples_colors'][n],
                                    **opts['plot_offdiag']
                                )
                            else:
                                pass

                    if len(points) > 0:

                        for n, v in enumerate(points):
                            h = plt.plot(
                                v[:, col], v[:, row],
                                color=opts['points_colors'][n],
                                **opts['points_offdiag']
                            )

        if len(subset) < dim:
            for row in range(len(subset)):
                ax = axes[row, len(subset)-1]
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                text_kwargs = {'fontsize': plt.rcParams['font.size']*2.}
                ax.text(x1 + (x1 - x0) / 8., (y0 + y1) / 2., '...', **text_kwargs)
                if row == len(subset)-1:
                    ax.text(x1 + (x1 - x0) / 12., y0 - (y1 - y0) / 1.5, '...', rotation=-45, **text_kwargs)

    return fig, axes
