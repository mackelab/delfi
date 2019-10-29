## **last**`#!py3 (ordered_dict)` { #last data-toc-label=last }


```
Returns last element of ordered dictionary
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def last(ordered_dict):
	    
	    key = next(reversed(ordered_dict))
	    return ordered_dict[key]
	
	```
