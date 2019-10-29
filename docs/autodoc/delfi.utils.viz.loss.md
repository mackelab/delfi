## **loss**`#!py3 (losses, key='trn', loss_clipping=1000.0, title='')` { #loss data-toc-label=loss }


```
Given an info dict, plot loss
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def loss(losses, key='trn', loss_clipping=1000., title=''):
	    
	
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
	
	```
