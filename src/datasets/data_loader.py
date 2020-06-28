import glob
import librosa
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import  Dataset
from torchvision import transforms, utils
import torch
import os
import torch.nn.functional as F
import cv2

def process_image(image_file,transforms):

	image_file = image_file.T
	# image = cv2.resize(image_file,(28,28))
	# print(image.shape)

	image = transforms(image_file)


	# print(image.size())

	return image

transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])


class read_dataset(Dataset):

	def __init__(self,file_path, labels_path, mode):
		
		self.df = pd.read_csv(labels_path)
		self.file_path = file_path	
		if(mode == "train"):

			self.transforms = transform_train
		else:
			self.transforms = transform_test


	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):


		image1_path = os.path.join(self.file_path,self.df["image_name"][index])
		image1_file = np.load(image1_path)

		# print("****************************************")	
		# print(np.max(image1_file))	
		# print(np.min(image1_file))	
		
		image1 = process_image(image1_file,self.transforms)

		# print(torch.max(image1))	
		# print(torch.min(image1))	
		
		label = self.df["labels"][index]
		
		image2_df = self.df.loc[self.df['labels'] == label]
		image2_path = os.path.join(self.file_path, image2_df.sample().values[0][0])
		image2_file = np.load(image2_path)
		image2 = process_image(image2_file,self.transforms)
		
		return (image1,image2, label)





