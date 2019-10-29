## **plot_hist_marginals**`#!py3 (data, lims=None, gt=None)` { #plot_hist_marginals data-toc-label=plot_hist_marginals }


```
Plots marginal histograms and pairwise scatter plots of a dataset
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def plot_hist_marginals(data, lims=None, gt=None):
	    
	    n_bins = int(np.sqrt(data.shape[0]))
	
	    if data.ndim == 1:
	        fig, ax = plt.subplots(1, 1, facecolor='white')
	        ax.hist(data, n_bins, normed=True)
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
	                    ax[i, j].hist(data[:, i], n_bins, normed=True)
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
	
	```
