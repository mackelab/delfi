## **samples_nd**`#!py3 (samples, points=[], **kwargs)` { #samples_nd data-toc-label=samples_nd }


```
Plot samples and points

See `opts` below for available keyword arguments.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def samples_nd(samples, points=[], **kwargs):
	    
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
	
	```
