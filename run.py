import os

from src.mafunction import add_encode_resultjson, clear_json_file, show_img, update_files_from_resultjson, validate_resultjson

#path = 'test/0507_02/img'
path = 'img'

list = []
# Scan all file.jpg in img/
for (root, dirs, file) in os.walk(path):
	for f in file:
		if '.jpg' in f:
			list.append(f)

clear_json_file('result')
for f in list:
	# cmd /k : remain after run
	# cmd /c : terminate after run
	os.system('cmd /c "python main.py -i img/%s"' %f)

	# In this step, in result.json, "encoded": ""
	# Add the encoded string to result.json for each image
	add_encode_resultjson(key_name=f)

# result.json will save all measurements result for each file in img/
# Then validate it, only remain RESULT key
#validate_resultjson()

# From final result.json, I need to update final.bin & final.jpg
#update_files_from_resultjson()

'''
import json, base64
with open('tmp/result.json', 'r') as fp:
	data = json.load(fp)
	byte = data['result']['encoded']

decodeit = open('imgcv/final.jpg', 'wb')
decodeit.write(base64.b64decode((byte)))
decodeit.close()
'''

#byte = tmp['encoded']
#decodeit = open('imgcv/final_0.jpg', 'wb')
#decodeit.write(base64.b64decode((byte)))
#decodeit.close()