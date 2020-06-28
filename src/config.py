import warnings
import numpy as np


class DefaultConfig(object):
	seed = 0

	train_images = "../data/training_data/train/images"
	train_labels = "../data/training_data/train/train.csv"
	val_images = "../data/training_data/val/images"
	val_labels = "../data/training_data/val/val.csv"
	

	# dataset options
	batch_size = 64
	image_size = 60
	num_channels = 3

	
	base_learning_rate = 0.0001
	style_dim = 512
	class_dim = 64

	
	style_dim = 1024
	class_dim = 512
	
	# style_dim = 16
	# class_dim = 16
	
	num_classes = 672

	
	reconstruction_coef = 2
	reverse_cycle_coef = 10
	kl_divergence_coef = 3
	
	beta1 = 0.9
	beta2 = 0.999



	start_epoch = 0
	total_epochs = 100


	use_gpu = True
	debug = True
	load_saved_model = False
	dataset = "sprites"	
	

	# model_name = 'specified_classifier' 
	model_name = 'vae' 
		
	save_dir = './pytorch-ckpt/sprites'



	def _state_dict(self):
		return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
				if not k.startswith('_')}

opt = DefaultConfig()





