## **first**`#!py3 (ordered_dict)` { #first data-toc-label=first }


```
Returns first element of ordered dictionary
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def first(ordered_dict):
	    
	    key = next(iter(ordered_dict))
	    return ordered_dict[key]
	
	```
