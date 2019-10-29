## **dist**`#!py3 (dist, title='')` { #dist data-toc-label=dist }


```
Given dist, plot histogram
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def dist(dist, title=''):
	    
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
	
	```
