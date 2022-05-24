import os


def printlog(msg='', var=None, isClearFirst=False, end='\n'):
	working_path = os.path.dirname(__file__)
	path = os.path.join(working_path, '../tmp/log.txt')
	
	if isClearFirst:
		with open(path, 'w'):
			pass

	with open(path, 'a') as f:
		if var == None:
			message = msg + end
		else:
			message = msg + ' ' + str(var) + end
		print(message,  file=f, end='')