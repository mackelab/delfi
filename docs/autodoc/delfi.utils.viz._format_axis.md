## **_format_axis**`#!py3 (ax, xhide=True, yhide=True, xlabel='', ylabel='', tickformatter=None)` { #_format_axis data-toc-label=_format_axis }



??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def _format_axis(ax, xhide=True, yhide=True, xlabel='', ylabel='',
	        tickformatter=None):
	    for loc in ['right', 'top', 'left', 'bottom']:
	        ax.spines[loc].set_visible(False)
	    if xhide:
	        ax.set_xlabel('')
	        ax.xaxis.set_ticks_position('none')
	        ax.xaxis.set_tick_params(labelbottom=False)
	    if yhide:
	        ax.set_ylabel('')
	        ax.yaxis.set_ticks_position('none')
	        ax.yaxis.set_tick_params(labelleft=False)
	    if not xhide:
	        ax.set_xlabel(xlabel)
	        ax.xaxis.set_ticks_position('bottom')
	        ax.xaxis.set_tick_params(labelbottom=True)
	        if tickformatter is not None:
	            ax.xaxis.set_major_formatter(tickformatter)
	        ax.spines['bottom'].set_visible(True)
	    if not yhide:
	        ax.set_ylabel(ylabel)
	        ax.yaxis.set_ticks_position('left')
	        ax.yaxis.set_tick_params(labelleft=True)
	        if tickformatter is not None:
	            ax.yaxis.set_major_formatter(tickformatter)
	        ax.spines['left'].set_visible(True)
	    return ax
	
	```
