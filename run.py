import os
from src.mafunction import printlog, get_path_relative_to_src, add_encode_resultjson, clear_json_file, update_files_from_resultjson, validate_resultjson, print_final_result


def run(isLinux=False):
	# Get the path of this file --> ~/fish-length-opencv/
	path = get_path_relative_to_src('../img/')

	list = []
	# Scan all file.jpg in img/
	for (root, dirs, file) in os.walk(path):
		for f in file:
			if '.jpg' in f:
				list.append(f)

	clear_json_file('result')
	
	printlog(isClearFirst=True)
	for f in list:
		# cmd /k : remain after run
		# cmd /c : terminate after run
		fpath = os.path.join(path, f)
		printlog( str(' [PROCESSING] %s ' %f).center(42, '='), end='\n\n')
		if isLinux: 
			os.system('python main.py --image %s' %fpath)
		else: 	# For Windows
			os.system('cmd /c "python main.py --image %s"' %fpath)

		# In this step, in result.json, "encoded": ""
		# Add the encoded string to result.json for each image
		add_encode_resultjson(key_name=f)
		printlog( str(' [DONE] %s ' %f).center(42, '='), end='\n\n\n' )

	# result.json will save all measurements result for each file in img/
	# Then validate it, only remain RESULT key
	validate_resultjson()

	# From final result.json, I need to update final.bin & final.jpg
	update_files_from_resultjson()

	# Print final result
	print_final_result()

if __name__ == '__main__':
	run(isLinux=False)