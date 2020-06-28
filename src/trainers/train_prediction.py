import os
import shutil
import sys
import time
from itertools import cycle
from os import path as osp
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.autograd import Variable

from utils.helpers import load_dataloaders, load_model,imshow_grid
from utils.logger import Logger
from utils.meters import AverageMeter
from itertools import cycle
from utils.loss import reparameterize
from pathlib import Path

	


class Train_prediction:
	def __init__(self, opt):
		
		self.opt = opt		
		torch.manual_seed(opt.seed)
		os.makedirs(opt.save_dir, exist_ok=True)

		print('=========user config==========')
		pprint(opt._state_dict())
		print('============end===============')

		self.trainloader, self.valloader = load_dataloaders(self.opt)
		
		self.use_gpu = opt.use_gpu
		self.device = torch.device('cuda')





		self._init_model()
		self._init_optimizer()
		self.criterion = nn.MSELoss()

		self.model_dir = os.path.join(self.opt.save_dir,str(self.opt.model_type) + "_"+str(self.opt.model_name))

		Path(self.model_dir).mkdir(parents=True, exist_ok=True)

		if not os.path.exists(os.path.join(self.model_dir, 'reconstructed_images')):
			os.makedirs(os.path.join(self.model_dir, 'reconstructed_images'))
			

		if(self.opt.debug == False):
			self.experiment = wandb.init(project="cycle_consistent_vae")
			hyper_params = self.opt._state_dict()
			self.experiment.config.update(hyper_params)
			wandb.watch(self.encoder)
			wandb.watch(self.prediction_model)



	def _init_model(self):

		"""
		model definition
		"""	
		self.encoder, self.decoder, self.prediction_model = load_model(self.opt)



		print("LEARNING RATE: ",self.opt.base_learning_rate)

		self.X = torch.FloatTensor(self.opt.batch_size, self.opt.num_channels, self.opt.image_size, self.opt.image_size)

		self.style_latent_space = torch.FloatTensor(self.opt.batch_size, self.opt.style_dim)


		if self.use_gpu:
			self.device = torch.device('cuda')

			self.encoder.cuda()
			self.prediction_model.cuda()

			self.decoder.cuda()

			self.X = self.X.cuda()

			self.style_latent_space = self.style_latent_space.cuda()


		self.load_encoder()
		self.load_decoder()

	def _init_optimizer(self):

		"""
		optimizer and scheduler definition
		"""
		self.optimizer = optim.Adam(self.prediction_model.parameters(),
									lr=self.opt.base_learning_rate,
									betas = (self.opt.beta1, self.opt.beta2))

		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=80, gamma=0.1)






	def train(self):
			print("TRAINING Prediction Model...")
			sys.stdout = Logger(osp.join(self.model_dir,'log_train.txt'))

			start_epoch = self.opt.start_epoch
			best_acc = 0.0
			best_epoch = 0

			for epoch in range(start_epoch, self.opt.total_epochs):
		
				print('')
				print('Epoch #' + str(epoch) + '..........................................................................')
				
				self.train_one_epoch(epoch)
				self.scheduler.step()

			if (epoch + 1) % 10 == 0 or (epoch + 1) == self.opt.total_epochs:

				self.evaluation(epoch)


				if(self.opt.debug == False):
					print("UPLOADING EVALUATION IMAGES ...")
					wandb.save(osp.join(self.model_dir,"reconstructed_images", str(epoch) + '_gt_image.png'))
					wandb.save(osp.join(self.model_dir,"reconstructed_images", str(epoch) + '_output_image.png'))
					


	
			self.save_model()
			if(self.opt.debug == False):

				print("UPLOADING FINAL FILES ...")
				wandb.save(self.model_dir + "/*")
				

				
			

	def train_one_epoch(self, epoch):


		self.prediction_model.train()
		
		self.mse_losses = AverageMeter()
		

	
		for batch_idx, data in enumerate(self.trainloader) :
			
			image_batch_1, image_batch_2, _ = data

			self.optimizer.zero_grad()
			self.X.copy_(image_batch_1)
			
			self.style_latent_space.normal_(0., 1.)
			
			style_mu, style_logvar, class_latent_space = self.encoder(Variable(self.X))
			style_latent_space = reparameterize(training = False, mu = style_mu, logvar = style_logvar)

			
			if(self.opt.model_type == "specified"):
	
				output = self.prediction_model(style_latent_space)

				self._compute_loss(class_latent_space, output)

			else:

				output = self.prediction_model(class_latent_space)
				self._compute_loss(style_latent_space, output)


			self.optimizer.step()



			del(image_batch_1)
			del(image_batch_2)

			# break
		self._print_values(epoch)


	def _compute_loss(self,inputs,outputs):

		loss = torch.sqrt(self.criterion(inputs, outputs))
		loss.backward()


		self.mse_losses.update(loss.item())
	


		
	def _print_values(self,epoch):

		if(self.opt.debug == False):
			self.experiment.log({'MSE Loss': self.mse_losses.mean},step = epoch)
			

		print('MSE Loss: ' + str(self.mse_losses.mean))


	def evaluation(self,epoch):
		self.load_prediction_model()
		self.prediction_model.eval()
		
		print("Evaluating Model ...")


		temp = iter(self.valloader)			
		data = temp.next()
		image_batch_1, image_batch_2, _ = data

		self.optimizer.zero_grad()
		self.X.copy_(image_batch_1)
		
		self.style_latent_space.normal_(0., 1.)
		
		style_mu, style_logvar, class_latent_space = self.encoder(Variable(self.X))
		style_latent_space = reparameterize(training = False, mu = style_mu, logvar = style_logvar)



		gt_recontruction = self.decoder(Variable(style_latent_space), class_latent_space.detach())
		if(self.opt.model_type == "specified"):
			
			output = self.prediction_model(style_latent_space)
			output_reconstruction = self.decoder(Variable(style_latent_space), output.detach())
	
		
		else:

			output = self.prediction_model(class_latent_space)
			output_reconstruction = self.decoder(Variable(output), class_latent_space.detach())



		gt_image_batch = (np.transpose(gt_recontruction.detach().cpu().numpy(), (0, 2, 3, 1)))	
		imshow_grid(gt_image_batch, name =str(epoch) + '_gt_image', save=True, save_dir = self.model_dir)

		
		output_image_batch = np.transpose(output_reconstruction.detach().cpu().data.numpy(), (0, 2, 3, 1))
		imshow_grid(output_image_batch, name = str(epoch) + '_output_image', save=True, save_dir = self.model_dir)


		


	def save_model(self):
		print("SAVING MODEL ...")
		torch.save(self.prediction_model.state_dict(), os.path.join(self.model_dir, "prediction.pth"))



	def load_encoder(self):

		print("[*] LOADING ENCODER: {}".format(os.path.join(self.opt.save_dir, "vae", "encoder.pth")))
		self.encoder.load_state_dict(torch.load(os.path.join(self.opt.save_dir, "vae", "encoder.pth")))

		self.encoder.cuda()


	def load_decoder(self):

		print("[*] LOADING DECODER: {}".format(os.path.join(self.opt.save_dir, "vae", "decoder.pth")))
		self.decoder.load_state_dict(torch.load(os.path.join(self.opt.save_dir, "vae", "decoder.pth")))
		self.decoder.cuda()
	
	def load_prediction_model(self):

		print("[*] LOADING PREDICTION MODEL: {}".format(os.path.join(self.model_dir, "prediction.pth")))
		self.prediction_model.load_state_dict(torch.load(os.path.join(self.model_dir, "prediction.pth")))
		self.prediction_model.cuda()
		


