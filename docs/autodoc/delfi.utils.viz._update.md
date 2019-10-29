## **_update**`#!py3 (d, u)` { #_update data-toc-label=_update }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
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
	
	```
