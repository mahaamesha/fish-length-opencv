import os

from src.mafunction import clear_json_file, decode_img, show_img, update_files_from_resultjson, validate_resultjson

#path = 'test/0507_02/img'
path = 'img'

list = []
# Scan all file.jpg in img/
for (root, dirs, file) in os.walk(path):
	for f in file:
		if '.jpg' in f:
			list.append(f)

print(list)

clear_json_file('result')
for f in list:
	# cmd /k : remain after run
	# cmd /c : terminate after run
	os.system('cmd /c "python main.py -i img/%s"' %f)
	print('cmd /c "python main.py -i img/%s"' %f)

validate_resultjson()
update_files_from_resultjson()