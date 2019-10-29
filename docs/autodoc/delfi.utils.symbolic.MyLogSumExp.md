## **MyLogSumExp**`#!py3 (x, axis=None)` { #MyLogSumExp data-toc-label=MyLogSumExp }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def MyLogSumExp(x, axis=None):
	    x_max = tt.max(x, axis=axis, keepdims=True)
	    return tt.log(tt.sum(tt.exp(x - x_max), axis=axis, keepdims=True)) + x_max
	
	```
