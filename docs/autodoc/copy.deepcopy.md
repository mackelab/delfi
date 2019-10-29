## **deepcopy**`#!py3 (x, memo=None, _nil=[])` { #deepcopy data-toc-label=deepcopy }


```
Deep copy operation on arbitrary Python objects.

See the module's __doc__ string for more info.
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def deepcopy(x, memo=None, _nil=[]):
	    
	
	    if memo is None:
	        memo = {}
	
	    d = id(x)
	    y = memo.get(d, _nil)
	    if y is not _nil:
	        return y
	
	    cls = type(x)
	
	    copier = _deepcopy_dispatch.get(cls)
	    if copier:
	        y = copier(x, memo)
	    else:
	        try:
	            issc = issubclass(cls, type)
	        except TypeError: # cls is not a class (old Boost; see SF #502085)
	            issc = 0
	        if issc:
	            y = _deepcopy_atomic(x, memo)
	        else:
	            copier = getattr(x, "__deepcopy__", None)
	            if copier:
	                y = copier(memo)
	            else:
	                reductor = dispatch_table.get(cls)
	                if reductor:
	                    rv = reductor(x)
	                else:
	                    reductor = getattr(x, "__reduce_ex__", None)
	                    if reductor:
	                        rv = reductor(4)
	                    else:
	                        reductor = getattr(x, "__reduce__", None)
	                        if reductor:
	                            rv = reductor()
	                        else:
	                            raise Error(
	                                "un(deep)copyable object of type %s" % cls)
	                if isinstance(rv, str):
	                    y = x
	                else:
	                    y = _reconstruct(x, memo, *rv)
	
	    # If is its own copy, don't memoize.
	    if y is not x:
	        memo[d] = y
	        _keep_alive(x, memo) # Make sure x lives at least as long as d
	    return y
	
	```
