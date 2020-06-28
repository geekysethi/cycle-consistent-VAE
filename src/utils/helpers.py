from __future__ import division

import errno
import os
from os import path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from datasets.data_loader import read_dataset
from models.baseline_new import Classifier, Decoder, Encoder, Prediction


def prepare_dirs(opt):

	print('==> Preparing data..')
	try:
		os.makedirs(os.path.join(opt.save_dir,opt.model_name))
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	if not os.path.exists(os.path.join(opt.save_dir,opt.model_name,'reconstructed_images')):
		os.makedirs(os.path.join(opt.save_dir, opt.model_name, 'reconstructed_images'))


def load_dataloaders(opt):
	print('[INFO] Initializing Dataset {}'.format(opt.dataset))

	
	train_set = read_dataset(opt.train_images, opt.train_labels, "train")
	train_loader = DataLoader(train_set, batch_size = opt.batch_size, shuffle=True, drop_last=True)
	
	
	val_set = read_dataset(opt.val_images, opt.val_labels, "test")
	val_loader =  DataLoader(val_set, batch_size = opt.batch_size, shuffle=True, drop_last=True)

	return train_loader, val_loader


def weights_init(layer):
	if isinstance(layer, nn.Conv2d):
		layer.weight.data.normal_(0.0, 0.05)
		layer.bias.data.zero_()
	elif isinstance(layer, nn.BatchNorm2d):
		layer.weight.data.normal_(1.0, 0.02)
		layer.bias.data.zero_()
	elif isinstance(layer, nn.Linear):
		layer.weight.data.normal_(0.0, 0.05)
		layer.bias.data.zero_()


def load_model(opt):

	print('Initializing Model ...')

	encoder = Encoder(style_dim = opt.style_dim, class_dim = opt.class_dim)
	encoder.apply(weights_init)


	decoder = Decoder(style_dim = opt.style_dim, class_dim = opt.class_dim)
	decoder.apply(weights_init)	



	if(opt.model_name == "vae"):
		print("LOADING VAE ...")
		return encoder, decoder


	elif(opt.model_name == "classifier"):

		if(opt.model_type == "specified"):
			print("LOADING SPECIFIED Classifier...")
			
			classifier = Classifier(opt.style_dim, opt.num_classes)

		else:
			print("LOADING UNSPECIFIED Classifier...")

			classifier = Classifier(opt.class_dim, opt.num_classes)

		return encoder, classifier


	elif(opt.model_name == "prediction"):


		
		if(opt.model_type == "specified"):
			print("LOADING SPECIFIED Prediction...")
			prediction = Prediction(opt.style_dim,opt.class_dim)

		else:
			print("LOADING UNSPECIFIED Prediction...")

			prediction = Prediction(opt.class_dim, opt.style_dim)


		return encoder, decoder, prediction







def imshow_grid(images, shape=[2, 8], name='default', save=False, save_dir = "./"):
	"""
	Plot images in a grid of a given shape.
	Initial code from: https://github.com/pumpikano/tf-dann/blob/master/utils.py
	"""
	fig = plt.figure(1)
	grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

	size = shape[0] * shape[1]
	for i in range(size):
		current_image = images[i]
		
		grid[i].axis('off')
		grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

	print("SAVING IMAGES")
	path = os.path.join(save_dir,'reconstructed_images', str(name) + '.png')
	plt.savefig(path)
	plt.clf()
	# else:
	#     plt.show()


# def imshow_grid(images, shape=[2, 8], name='default', save=False,opt = "./"):
# 	"""
# 	Plot images in a grid of a given shape.
# 	Initial code from: https://github.com/pumpikano/tf-dann/blob/master/utils.py
# 	"""

# 	current_image = images[0] * 255
# 	current_image = current_image.astype("uint8")
# 	path = os.path.join(opt.save_dir,opt.model_name,'reconstructed_images', str(name) + '.png')

# 	cv2.imwrite(path,current_image)




def save_images(original_batch,style_batch,reconstructed_batch,opt):

	count = 0
	for i in range(len(original_batch)):

		print(i)
		image_name = "{}.png".format(str(count).zfill(8))

		original_image = original_batch[i] * 255
		original_image = original_image.astype("uint8")
		
		style_image = style_batch[i]*255
		style_image = style_image.astype("uint8")

		reconstructed_image = reconstructed_batch[i] * 255
		reconstructed_image = reconstructed_image.astype("uint8")

		cv2.imwrite(os.path.join(opt.save_dir,"output_images","original",image_name),original_image)
		cv2.imwrite(os.path.join(opt.save_dir,"output_images","style",image_name),style_image)
		cv2.imwrite(os.path.join(opt.save_dir,"output_images","reconstructed",image_name),reconstructed_image)
		count+=1
		# break




def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


def slerpolate(x, y, C, num_pts):
    alphas = np.linspace(0, 1.0, num_pts)
    if C is not None:
        a = np.linalg.cholesky(C)
        a_inv = np.linalg.inv(a)
        x_inv = np.dot(a_inv, x)
        y_inv = np.dot(a_inv, y)
        res_inv = [slerp(alpha, x_inv, y_inv) for alpha in alphas]
        res = np.dot(a, np.array(res_inv).T)
        # res = np.transpose(res_inv)
        return res
    else:
        return np.array([slerp(alpha, x, y) for alpha in alphas]).T
