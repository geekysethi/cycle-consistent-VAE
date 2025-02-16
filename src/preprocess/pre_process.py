import scipy.io
import numpy as np
import h5py 
import os
import sys
import cv2
import pandas as pd

def read_mat_file(path):
	f = h5py.File(path,'r') 
	return f
	
	
def save_image(image,count,mode):
	image_name = "{}.npy".format(str(count).zfill(8))
	save_path = os.path.join(training_path,mode,"images",image_name)
	print(save_path)
	np.save(save_path, image)

	return image_name

	

def process_img(h5py_dic,mode):

	global count
	image_name_list = []

	a_group_key = list(h5py_dic.keys())[3]
	
	data = list(h5py_dic[a_group_key])
	for i in range(len(data)):
		print("*"*40)
		current_row = h5py_dic[data[i][0]][()]
		print(current_row.shape)
		for current_image in current_row:				
			current_image = np.array(current_image)
			current_image = current_image.reshape(3,60,60)
			image_name = save_image(current_image,count,mode)			
			image_name_list.append(image_name)
			count += 1
			# break
		# break

	return image_name_list


def process_vector(h5py_dic):
	a_group_key = list(h5py_dic.keys())[1]
	
	data = list(h5py_dic[a_group_key])
	# print(data)
	vector = np.concatenate(data, axis=0)
	
	return np.array(vector)



def split_data(ids,mode):

	image_list = []
	vector_list = []
	id_list = [] 
	for current_id in sorted(ids):
		current_id -= 1
		# print(current_id)

		current_mat_file_path = os.path.join(data_path,"sprites_"+str(current_id)+".mat") 
		print(current_mat_file_path)
		h5py_dic = read_mat_file(current_mat_file_path)
		
		current_image_list = process_img(h5py_dic,mode)

		current_vector = process_vector(h5py_dic)
		
		current_vector_list = [list(current_vector)] * len(current_image_list)
		current_id_list = [current_id] * len(current_image_list)

		image_list.extend(current_image_list)
		vector_list.extend(current_vector_list)
		id_list.extend(current_id_list)
		
		
		# break
	

	print("[INFO] SAVING DATA IN DATAFRAME")
	df = pd.DataFrame({"image_name":image_list,"vector":vector_list,"labels":id_list})
	
	df.to_csv(os.path.join(training_path,mode,str(mode) + ".csv"), encoding="utf-8", index=False)


if __name__ == "__main__":
	global count
	count = 0

	train_val_split = scipy.io.loadmat('../../data/sprites/sprites_splits.mat')
	train_ids = train_val_split["trainidx"][0]
	val_ids = train_val_split["validx"][0]
	data_path = "../../data/sprites/"
	training_path = "../../data/training_data"
	split_data(train_ids,"train")
	# split_data(val_ids,"val")
	

	# print((sorted(train_ids)))
	# print(val_ids)
	



