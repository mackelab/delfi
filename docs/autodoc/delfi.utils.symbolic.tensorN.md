## **tensorN**`#!py3 (N, name=None, dtype='float64')` { #tensorN data-toc-label=tensorN }


```
Return a tensor of the specified dimension.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def tensorN(N, name=None, dtype=theano.config.floatX):
	    
	    if N == 1:
	        return tt.vector(name=name, dtype=dtype)
	    if N == 2:
	        return tt.matrix(name=name, dtype=dtype)
	    elif N == 3:
	        return tt.tensor3(name=name, dtype=dtype)
	    elif N == 4:
	        return tt.tensor4(name=name, dtype=dtype)
	    else:
	        raise NotImplementedError
	
	```
