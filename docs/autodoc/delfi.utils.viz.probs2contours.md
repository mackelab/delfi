## **probs2contours**`#!py3 (probs, levels)` { #probs2contours data-toc-label=probs2contours }


```
Takes an array of probabilities and produces an array of contours at specified percentile levels

Parameters
----------
probs : array
    Probability array. doesn't have to sum to 1, but it is assumed it contains all the mass
levels : list
    Percentile levels, have to be in [0.0, 1.0]

Return
------
Array of same shape as probs with percentile labels
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def probs2contours(probs, levels):
	    
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
	
	```
