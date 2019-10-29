## **nth**`#!py3 (ordered_dict, n)` { #nth data-toc-label=nth }


```
Returns nth element of ordered dictionary
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def nth(ordered_dict, n):
	    
	    key_val = list(ordered_dict.items())[n]
	    return key_val[1]
	
	```
