import warnings
import numpy as np


class DefaultConfig(object):
	
	seed = 0

	train_images = "../data/classification_data/images"
	train_labels = "../data/classification_data/train.csv"
	val_images = "../data/classification_data/images"
	val_labels = "../data/classification_data/val.csv"
	

	# dataset options
	batch_size = 256
	image_size = 60
	num_channels = 3

	
	base_learning_rate = 0.001
	style_dim = 512
	class_dim = 64
	
	# style_dim = 16
	# class_dim = 16
	
	style_dim = 1024
	class_dim = 512

	num_classes = 672

	

	beta1 = 0.9
	beta2 = 0.999



	start_epoch = 0
	total_epochs = 100


	use_gpu = True
	debug = True
	dataset = "sprites"	
	

	# model_name = 'specified_classifier' 
	model_name = 'classifier' 
	model_type = "specified"   #specified or unspecified
	save_dir = './pytorch-ckpt/sprites'



	def _state_dict(self):
		return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
				if not k.startswith('_')}

opt = DefaultConfig()





