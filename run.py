import os, json

#path = 'test/0507_02/img'
path = 'img'

list = []
# Scan all file.jpg in img/
for (root, dirs, file) in os.walk(path):
	for f in file:
		if '.jpg' in f:
			list.append(f)

print(list)

for f in list:
	# cmd /k : remain after run
	# cmd /c : terminate after run
	os.system('cmd /c "python main.py -i img/%s"' %f)
	print('cmd /c "python main.py -i img/%s"' %f)

	#tmp_num = []
	#tmp_length = []
	#tmp_encod = []
	#print('tampung nilai num_fish & fishlength & encoded_string')	# ada 5 data untuk setiap list
	
	# Create tmp.json
	# Access data from result.json in fish-length-opencv
	# Append the dict into tmp.json