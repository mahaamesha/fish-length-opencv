import os


def printlog(msg, var=None, isClearFirst=False):
	working_path = os.path.dirname(__file__)
	path = os.path.join(working_path, 'log.txt')
	
	if isClearFirst:
		with open(path, 'w'):
			pass

	with open(path, 'a') as f:
		if var == None:
			message = msg
		else:
			message = msg + ' ' + str(var)
		print(message,  file=f)


num = 222

printlog('asdad', num, isClearFirst=True)
printlog('kutaii')

