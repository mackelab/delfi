## **info**`#!py3 (info, html=False, title=None)` { #info data-toc-label=info }


```
Given info dict, produce info text
```


??? info "Source Code" 
	```py3 linenums="1 1 2" 
	def info(info, html=False, title=None):
	    
	    if title is None:
	        infotext = u''
	    else:
	        if html:
	            infotext = u'<b>{}</b><br>'.format(title)
	        else:
	            infotext = u'{}\n'.format(title)
	
	    for key, value in info.items():
	        if key not in ['losses']:
	            infotext += u'{} : {}'.format(key, value)
	            if html:
	                infotext += '<br>'
	            else:
	                infotext += '\n'
	
	    return infotext
	
	```
