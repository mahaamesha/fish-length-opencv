import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os
import base64
from datetime import datetime

import src.segmentation as s
import src.command as cmd


def get_path_relative_to_src(path2=''):
	# Fill the arguments with filename or dir/filename: 
	# 'filename.format' or 'directory/filename.format'
	# Relative from this file path --> src/ folder
	running_file_path = os.path.dirname(__file__)
	if path2 == '':
		return running_file_path
	else:
		path = os.path.join(running_file_path, path2)
		return path


def only_last_path(full_path='srcpath\../dirname/filename.format'):
	# Use full_path from get_path_relative_to_src()
	limit_slash = 2
	for i in range(len(full_path)-1, -1, -1):
		if full_path[i] == '/':
			limit_slash -= 1
			if limit_slash == 0:
				return str(full_path[i+1:])


def is_file_empty(file_path):
	""" Check if file is empty by confirming if its size is 0 bytes"""
	return os.path.exists(file_path) and os.stat(file_path).st_size == 0


def clear_json_file(title):
	filename = title + '.json'
	path = get_path_relative_to_src('../tmp/') + filename
	with open(path,'w'):
		pass


def write_json(title, dict):
	filename = title + '.json'
	path = get_path_relative_to_src('../tmp/') + filename

	with open(path,'r+') as file:
		file_data = json.load(file)
		key = file_data.keys()
		file_data[key].append(dict)
		file.seek(0)
		json.dump(file_data, file, indent=4)


def generate_imagesjson(list_img=s.list_img):
	path = get_path_relative_to_src('../tmp/images.json')
	with open(path, 'a'):
		pass
	with open(path, 'w') as f:
		f.write('{\n')
		num_row = len(list_img)
		for i in range(num_row):
			f.write('\t"' + list_img[i][2].upper() + '": {\n')
			f.write('\t\t"_id": ' + str(i) + ',\n')
			f.write('\t\t"_var": "' + list_img[i][2] + '",\n')
			f.write('\t\t"_showflag": ' + str(list_img[i][0]) + ',\n')
			f.write('\t\t"_encodeflag": ' + str(list_img[i][1]) + '\n')
			if i != num_row-1: f.write('\t},\n')
			else: f.write('\t}\n')
		f.write('}')
	print( str('Generate images.json').ljust(37,'.') + str('Done').rjust(5,' '), end='\n\n')

def points2json(title, dict):
	# dict = {
	#   'x': [...], 
	#   'y': [...]
	# }
	# final = {
	#   'curve_0':{
	#       'x': [...],
	#       'y': [...]
	#   }, ...
	filename = title + '.json'
	path = get_path_relative_to_src('../tmp/') + filename

	with open(path, 'r') as fr:
		if (not is_file_empty(path)):
			final_dict = json.load(fr)      # points.json should be initialized first & not empty
		else: final_dict = {}
	
		# if more than num_of_contours, delete old data
		if (len(final_dict) > len(s.cnts)):
			for i in range(len(final_dict)-1):
				del final_dict[str('curve_'+str(i))]
	final_dict[str('curve_'+str(len(final_dict)))] = dict

	with open(path, 'w') as fp:
		json.dump(final_dict, fp, indent=4)
	#print('Export', filename, '... Done')


# check images.json, return '_id'
def get_id_imgjson(_var):    # input string, ex: 'images'
	path = get_path_relative_to_src('../tmp/images.json')
	with open(path, 'r') as fp:
		data = json.load(fp)
		for key in data.keys():
			var = data[key]['_var']
			_id = data[key]['_id']

			if (var == _var): 
				return int(_id)


def show_img(title, img, destroy_all=False):
	cv.imshow(title, img)
	cv.waitKey(0)
	if destroy_all == True:
		cv.destroyAllWindows()
	print( str('\tShow ' + title + '.jpg').ljust(30,'.') + str('Done').rjust(5,' ') )


# to not show, set the '_flag' to 0 in images.json
def show_imgjson():
	print(str('Show selected img listed in images.json:'))
	path = get_path_relative_to_src('../tmp/images.json')
	with open(path, 'r') as fp:
		data = json.load(fp)

		for val in data.values():
			if val['_showflag']:
				show_img(val['_var'], s.list_img[ val['_id'] ][3])
	print()
		

def save_img(title, img):
	filename = title + '.jpg'
	path = get_path_relative_to_src('../imgcv/') + filename
	status = cv.imwrite(path, img)
	# folder should be initialized first
	
	print( str('\t' + only_last_path(path)).ljust(30,'.') + str(status).rjust(5,' ') )


# save all to imgcv, source from images.json
# to improve memory, I only save img with _encodeflag == 1
def save_imgjson():
	print('Save all img listed in images.json:')
	path = get_path_relative_to_src('../tmp/images.json')
	with open(path, 'r') as fp:
		data = json.load(fp)
		for val in data.values():
			if val['_encodeflag'] == 1:
				save_img(val['_var'], s.list_img[ val['_id'] ][3])
	print()


# img to string
def encode_img(title='final'):
	path_imgcv = get_path_relative_to_src('../imgcv/') + title + '.jpg'
	path_bin = get_path_relative_to_src('../bin/') + title + '.bin'

	with open(path_imgcv, 'rb') as image2string:
		converted_string = base64.b64encode(image2string.read())
	
	with open(path_bin, 'wb') as file:
		file.write(converted_string)
	
	print(str('\t' + only_last_path(path_bin)).ljust(30,'.') + str('Done').rjust(5,' '))
	return str(converted_string)
	

# string to img
def decode_img(title='final'):  # check filename in folder imgcv
	path_imgcv = get_path_relative_to_src('../imgcv/') + title + '.jpg'
	path_bin = get_path_relative_to_src('../bin/') + title + '.bin'

	file = open(path_bin, 'rb')
	byte = file.read()
	file.close()
	
	decodeit = open(path_imgcv, 'wb')
	decodeit.write(base64.b64decode((byte)))
	decodeit.close()
	
	print( str('\t' + only_last_path(path_imgcv)).ljust(30,'.') + str('Done').rjust(5,' ') )


# save img string to json only if _encodflag == 1
def encode_imgjson():
	print('Encode *.jpg in /imgcv/:')
	path = get_path_relative_to_src('../tmp/images.json')
	with open(path, 'r') as fp:
		data = json.load(fp)
		for val in data.values():
			if val["_encodeflag"] == 1:     # change _encodeflag in segmentation.py
				filename = val["_var"]
				encode_img(filename)	# save to  filename.bin

	with open(path, 'w') as fp:
		json.dump(data, fp, indent=4)   # write images.json
	print()


def decode_imgjson():
	print('Decode *.bin in /bin/:')
	path = get_path_relative_to_src('../tmp/images.json')
	with open(path, 'r') as fp:
		data = json.load(fp)
		for val in data.values():
			if val["_encodeflag"] == 1:
				decode_img( str(val['_var']))
	print()


def add_encode_resultjson(key_name='0.jpg'):
	path = get_path_relative_to_src('../tmp/result.json')
	with open(path, 'r') as fp:
		data = json.load(fp)

		# Load the encoded string from bin file & input it to ENCODED key
		path = get_path_relative_to_src('../bin/final.bin')
		with open(path, 'r') as bfp:
			encoded_str = bfp.read()
			data[key_name]['encoded'] = str(encoded_str)
	
	path = get_path_relative_to_src('../tmp/result.json')
	with open(path, 'w') as fp:
		json.dump(data, fp, indent=4)


def calc_perimeter(cnts=s.cnts):
	perimeter = []
	for i in cnts:
		arc_length = cv.arcLength(i, True)
		perimeter.append(arc_length)
	print("perimeter:", "{:.2f}".format(perimeter))

	return perimeter


def calc_area(cnts=s.cnts):
	area = []
	for cnt in cnts:
		area.append( cv.contourArea(cnt) )
	#print("area:", area)

	return area


def num_moments(cnts=s.cnts):
	moments = []
	for cnt in cnts:
		moments.append( cv.moments(cnt) )
	print("moment:", moments)

	return moments


def get_skeleton(img=s.erodila):
	# Draw skeleton of banana on the mask
	size = np.size(img)
	skel = np.zeros(img.shape,np.uint8)
	ret, img = cv.threshold(img,5,255,0)
	element = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
	done = False
	while( not done):
		eroded = cv.erode(img,element)
		temp = cv.dilate(eroded,element)
		temp = cv.subtract(img,temp)
		skel = cv.bitwise_or(skel,temp)
		img = eroded.copy() 
		zeros = size - cv.countNonZero(img)
		if zeros==size: done = True
	kernel = np.ones((2,2), np.uint8)
	skel = cv.dilate(skel, kernel, iterations=1)
	skeleton_contours, _ = cv.findContours(skel, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	#import imutils
	#skeleton_contours = imutils.grab_contours(skeleton_contours)
	#largest_skeleton_contour = max(skeleton_contours, key=cv.contourArea)

	cv.drawContours(img, skel, -1, (0, 0, 255), 2)
	show_img("skeleton", skel, False)


def fit_line(cnts=s.cnts, image=s.image):
	for cnt in cnts:
		rows, cols = image.shape[:2]
		[vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
		lefty = int((-x*vy/vx) + y)
		righty = int(((cols-x)*vy/vx)+y)
		cv.line(image, (cols-1,righty),(0,lefty),(0,255,0),2)
	show_img("fit", image, False)


# To draw coordinate for all edge points
approx_factor = 5 * 1/1000  # More smaller, more points detected
def edge_points(cnts=s.cnts, image=s.image):
	font = cv.FONT_HERSHEY_COMPLEX

	# Going through every contours found in the image.
	for cnt in cnts :
		approx = cv.approxPolyDP(cnt, approx_factor * cv.arcLength(cnt, True), True)

		# draws boundary of contours.
		cv.drawContours(image, [approx], 0, (0, 0, 255), 5) 
	
		# Flaten the array
		n = approx.ravel()
		#print("Flaten array:", n)   # For future: store in json file
		i = 0
	
		for j in n :
			if(i % 2 == 0):
				x = n[i]
				y = n[i + 1]
	
				# String containing the co-ordinates.
				string = str(x) + " " + str(y) 
	
				if(i == 0):
					# text on topmost co-ordinate.
					cv.putText(image, "Arrow tip", (x, y), font, 0.5, (255, 0, 0)) 
				else:
					# text on remaining co-ordinates.
					cv.putText(image, string, (x, y), font, 0.5, (0, 255, 0)) 
			i = i + 1


# Length of fitting curve result. dict_points --> points in fit_poly()
def curve_length(dict_points):  # distance of 2 points: A and B
	sum = 0
	length = len( dict_points['x'] )
	for i in range(length-1):
		A = ( dict_points['x'][i], dict_points['y'][i] )
		B = ( dict_points['x'][i+1], dict_points['y'][i+1] )
		dx = abs(B[0] - A[0])   # |x2-x1|
		dy = abs(B[1] - A[1])   # |y2-y1|

		dl = math.sqrt(dy*dy + dx*dx)

		sum += dl
	
	#print('Fish Length:', "{:.2f}".format(sum))
	return sum


# measure fish length with data from points.json
def get_fish_length():
	fish_length = {}
	path = get_path_relative_to_src('../tmp/points.json')
	with open(path, 'r') as fp:
		data = json.load(fp)
		for key in data.keys(): # key are "curve_0", "curve_1", ...
			A = fish_length[str('fish_'+ key[6:])] = {}
			A.update( {'length': curve_length(data[key])} )
	
	# Write it to fish_length.json
	path = get_path_relative_to_src('../tmp/fish_length.json')
	with open(path, 'w') as fp:
		json.dump(fish_length, fp, indent=4)

	print( str('Measure fish length').ljust(37,'.') + str('Done').rjust(5,' '), end='\n\n')


# This will be better if len(data) > 1
def validate_data(data):
	copy = data[:]
	err = 0.3
	#print('data:', data)
	for n in data:
		avg = np.average(copy)
		lower = avg - err*avg
		upper = avg + err*avg
		#print(avg, lower, upper)
		
		# remove small data
		if n < lower:
			copy.remove(n)
		# handle large data
		elif n > upper:
			factor = int(round(n / avg, 0))
			copy.remove(n)
			for i in range(factor):
				avg = np.average(copy)
				copy.append(avg)
		#print('copy:', copy)
	data = copy[:]
	return data


# This will be better if len(data) > 1
def validate_num_fish():
	area = calc_area(s.cnts)    # array of floats
	area = validate_data(area)
	return len(area)


# This will be better if len(data) > 1
def validate_fishlength():
	fishlength = []
	fish_length = {}
	
	path = get_path_relative_to_src('../tmp/fish_length.json')
	with open(path, 'r') as fp:
		data = json.load(fp)
		
		for val in data.values():
			fishlength.append(val['length'])       
		fishlength = validate_data(fishlength)

		i = 0
		for val in data.values():
			A = fish_length[str('fish_' + str(i))] = {}
			A.update( {'length': fishlength[i]} )
			i += 1

	# Revision fish_length.json
	clear_json_file('fish_length')
	with open(path, 'w') as fp:
		json.dump(fish_length, fp, indent=4)
	
	return np.average(fishlength)


def append_by_key_resultjson(measurement='avr_fishlength'):
	# key: datetime, num_fish, avg_fishlength, encoded
	path = get_path_relative_to_src('../tmp/result.json')
	arr = []
	with open(path, 'r') as fp:
		data = json.load(fp)
		for key in data.keys():
			if key != 'result':
				arr.append( data[key][measurement] )
	return arr


# Last validation process & update the RESULT key
def validate_resultjson():
	# Eliminate noise
	result_avg_fishlength = append_by_key_resultjson('avg_fishlength')
	result_avg_fishlength = validate_data(result_avg_fishlength)

	# Choose the best data
	avg = np.average(result_avg_fishlength)
	nearest = result_avg_fishlength[0]
	best = result_avg_fishlength[0]
	for n in result_avg_fishlength:
		if abs(n - avg) < nearest:
			nearest = abs(n - avg)
			best = n
	
	# Check the data in result.json and save into RESULT key
	tmp = {}
	path = get_path_relative_to_src('../tmp/result.json')
	with open(path, 'r') as fp:
		data = json.load(fp)
		for key in data.keys():
			if data[key]['avg_fishlength'] == best:
				tmp = data[key]
		data = {'result': tmp.copy()}
				
	with open(path, 'w') as fp:
		json.dump(data, fp, indent=4)


# To update final.bin & final.jpg
def update_files_from_resultjson():
	path = get_path_relative_to_src('../tmp/result.json')
	# Copy encoded string from ENCODED key into new_encoded
	with open(path, 'r') as fp:
		data = json.load(fp)
		new_encoded = (data['result']['encoded'])
		data['result']['encoded'] = 'bin/final.bin'		# To reduce memory size
	with open(path, 'w') as fp:
		json.dump(data, fp, indent=4)

	# Save encoded string from new_encoded into final.bin
	path = get_path_relative_to_src('../bin/final.bin')
	with open(path, 'w') as bfp:
		bfp.write(new_encoded)
	
	# Use final.bin to decode img. It will be saved as final.jpg in imgcv/
	decode_imgjson()


points = {}
def fit_poly(cnts=s.cnts, showPlot=False, option=1):
	# option to choose: 1 more points used | 0 fewer points used
	# Get all approx points in a contour
	# approx_factor defined in 'edge_points()'
	
	clear_json_file('points')
	for cnt in cnts :
		# ravel() to flatten numpy array
		if (option == 0):
			approx = cv.approxPolyDP(cnt, approx_factor * cv.arcLength(cnt, True), True)   # 0.005 : more lower more points detected
			n = approx.ravel()
		else: n = cnt.ravel()

		# Extract x, y coordinate
		x = []
		y = []
		for i in range(len(n)):
			# Get x and y
			if (i % 2 == 0):
				x.append(n[i])
			else:
				y.append(n[i])
		
		mymodel = np.poly1d(np.polyfit(x, y, 3))    # 3th degree polynomial
		myline = np.linspace(min(x), max(x), 20)    # step 20 default

		# get all curve fitting coordinate
		list_x = myline.tolist()
		list_y = []
		for i in list_x:
			list_y.append( mymodel(i) )
		points["x"] = list_x
		points["y"] = list_y

		# Export points to json
		points2json('points', points)

		if showPlot:
			plt.scatter(x, y)
			plt.plot(myline, mymodel(list_x))
			plt.show()
	print( str('Curve fitting for every contour').ljust(37,'.') + str('Done').rjust(5,' '), end='\n\n')


def avg_fishlength():
	path = get_path_relative_to_src('../tmp/fish_length.json')
	with open(path, 'r') as fp:
		data = json.load(fp)

		sum = 0
		for val in data.values():
			sum += val['length'] 
			#print(val['length'])
		avg = sum / len(data)
		#print('avg', avg)
	return avg


def generate_resultjson():
	path = get_path_relative_to_src('../tmp/result.json')
	with open(path, 'a'):
			pass
	if is_file_empty(path):
		with open(path, 'w') as f:
			f.write('{\n')
			f.write('\t"result": \n\t{\n')
			f.write('\t\t"datetime": "edit_later",\n')
			f.write('\t\t"num_fish": "edit_later",\n')
			f.write('\t\t"avg_fishlength": "edit_later",\n')
			f.write('\t\t"encoded": "edit_later"\n')
			f.write('\t}\n}')
	
	with open(path, 'r') as fp:
		data = json.load(fp)

		# Variable that will be saved temporarily
		now = datetime.now()
		now = now.strftime("%m/%d/%Y %H:%M:%S")
		num = validate_num_fish()
		avg = avg_fishlength() 
		
		# Temporary dictionary
		tmp = {
			"datetime": now,
			"num_fish": num,
			"avg_fishlength": avg,
			"encoded": ''	# later, in f.encode_imgjson()
		}

		# Create new key based on img path
		#key_name = str(cmd.args["image"]).replace('img/', '')
		key_name = str( only_last_path(full_path=cmd.args["image"]) ).replace('img/', '')
		data[key_name] = tmp
	
	# Write the json file
	with open(path, 'w') as fp:
		json.dump(data, fp, indent=4)


def get_info_resultjson(info='datetime'):
	# info = 'datetime' | 'num_fish' | 'avg_fishlengh'
	path = get_path_relative_to_src('../tmp/result.json')
	with open(path, 'r') as fp:
		data = json.load(fp)
		for key in data.keys():
			#key_name = str(cmd.args['image']).replace('img/', '')
			key_name = str( only_last_path(full_path=cmd.args["image"]) ).replace('img/', '')
			if key == key_name:
				if info == 'avg_fishlength':
					return str( round(data[key][info],2) )
				else:
					return str(data[key][info])
			

def plot_curve2img(title='final.jpg', showPlot=False):
	plt.rcParams["figure.autolayout"] = True

	path = get_path_relative_to_src('../imgcv/') + title
	im = plt.imread(path)
	fig, ax = plt.subplots()
	im = ax.imshow(im)

	path = get_path_relative_to_src('../tmp/points.json')
	with open(path, 'r') as fp:
		data = json.load(fp)
		
		for key in data.keys():     # key: curve_0, curve_1, ...
			# data['x] have been sorted automaticaly
			x = data[key]['x']
			y = data[key]['y']
			
			ax.plot(x, y, ls='dotted', linewidth=5, color='red')

	
	# Add title: datetime, num_of_fish, avg_fishlength
	dtt = get_info_resultjson('datetime')
	num = get_info_resultjson('num_fish') + ' fish'
	avg = get_info_resultjson('avg_fishlength') + ' mm'
	text_str = str(dtt + ' | ' + num + ' | ' + avg)
	ax.set_title(text_str)

	if showPlot: plt.show()
	path = get_path_relative_to_src('../imgcv/final.jpg')
	plt.savefig(path)
	s.final = cv.imread(path)
	print( str('Plot curve to original img').ljust(37,'.') + str('Done').rjust(5,' '), end='\n\n')


def numbering_curve():
	path = get_path_relative_to_src('../tmp/points.json')
	with open(path, 'r') as fp:
		data = json.load(fp)

		num = 0
		for val in data.values():
			num += 1
			
			x = int( np.average(val['x']) )
			y = int( np.average(val['y']) )

			text = '#' + str(num)
			cv.putText(s.final, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
	cv.imwrite('imgcv/final.jpg', s.final)


def print_final_result():
	path = get_path_relative_to_src('../tmp/result.json')
	with open(path, 'r') as f:
		data = json.load(f)
		for val in data.values():
			arr_key = list(val.keys())
			arr_val = list(val.values())
		
		print()
		print( str(' FINAL RESULT ').center(42, '='), end='\n\n' )
		for i in range(len(arr_key)):
			print( str(arr_key[i]).ljust(17, ' ') + str(': %s' %arr_val[i]))
		print()
		print( str(' END ').center(42, '='), end='\n\n' )
		