import os
import shutil
import sys
import time
from itertools import cycle
from os import path as osp
from pprint import pprint

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.autograd import Variable

from utils.helpers import (imshow_grid, load_dataloaders, load_model,
						   save_images, slerpolate)
from utils.logger import Logger
from utils.loss import l1_loss, mse_loss, reparameterize
from utils.meters import AverageMeter


class Train_vae:
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

		self.trainloader_iter = cycle(self.trainloader)
		self.valloader_iter = cycle(self.valloader)
		



		self._init_model()
		self._init_criterion()
		self._init_optimizer()

		self.save_path = os.path.join(self.opt.save_dir,self.opt.model_name)

		
		if(self.opt.debug == False):
			self.experiment = wandb.init(project="cycle_consistent_vae")
			hyper_params = self.opt._state_dict()
			self.experiment.config.update(hyper_params)
			wandb.watch(self.encoder)
			wandb.watch(self.decoder)



	def _init_model(self):

		"""
		model definition
		"""	

		self.encoder, self.decoder = load_model(self.opt)
		
				
		print("LEARNING RATE: ",self.opt.base_learning_rate)

		self.X_1 = torch.FloatTensor(self.opt.batch_size, self.opt.num_channels, self.opt.image_size, self.opt.image_size)
		self.X_2 = torch.FloatTensor(self.opt.batch_size, self.opt.num_channels, self.opt.image_size, self.opt.image_size)
		self.X_3 = torch.FloatTensor(self.opt.batch_size, self.opt.num_channels, self.opt.image_size, self.opt.image_size)

		self.style_latent_space = torch.FloatTensor(self.opt.batch_size, self.opt.style_dim)


		if self.use_gpu:

			self.encoder.cuda()
			self.decoder.cuda()


			self.device = torch.device('cuda')

			self.X_1 = self.X_1.cuda()
			self.X_2 = self.X_2.cuda()
			self.X_3 = self.X_3.cuda()

			self.style_latent_space = self.style_latent_space.cuda()

		if(self.opt.load_saved_model):

			self.load_model()



	
	def _init_optimizer(self):

		"""
		optimizer and scheduler definition
		"""
		self.auto_encoder_optimizer = optim.Adam(
			list(self.encoder.parameters()) + list(self.decoder.parameters()),
			lr=self.opt.base_learning_rate,
			betas = (self.opt.beta1, self.opt.beta2))

		self.reverse_cycle_optimizer = optim.Adam(
			list(self.encoder.parameters()),
			lr = self.opt.base_learning_rate,
			betas=(self.opt.beta1, self.opt.beta2)
		)

		# divide the learning rate by a factor of 10 after 80 epochs
		self.auto_encoder_scheduler = optim.lr_scheduler.StepLR(self.auto_encoder_optimizer, step_size=80, gamma=0.1)
		self.reverse_cycle_scheduler = optim.lr_scheduler.StepLR(self.reverse_cycle_optimizer, step_size=80, gamma=0.1)




	def _init_criterion(self):

		"""
		loss definitions
		"""
		self.cross_entropy_loss = nn.CrossEntropyLoss()
		self.cross_entropy_loss.cuda()








	def train(self):
		sys.stdout = Logger(osp.join(self.opt.save_dir, self.opt.model_name,'log_train.txt'))
		print("TRAINING VAE ...")
	
		"""
		variable definition
		"""
	

		start_epoch = self.opt.start_epoch
		best_acc = 0.0
		best_epoch = 0
			
		for epoch in range(start_epoch, self.opt.total_epochs):

			
			print('')
			print('Epoch #' + str(epoch) + '..........................................................................')
			
			self.train_one_epoch(epoch)
			# break


			self.auto_encoder_scheduler.step()
			self.reverse_cycle_scheduler.step()


			if (epoch + 1) % 20 == 0 or (epoch + 1) == self.opt.total_epochs:

				self.evaluation(epoch)
				self.save_model(epoch)

				if(self.opt.debug == False):
					print("UPLOADING FINAL FILES ...")
					wandb.save(osp.join(self.opt.save_dir,self.opt.model_name,"reconstructed_images", str(epoch) + '_original.png'))
					wandb.save(osp.join(self.opt.save_dir,self.opt.model_name, "reconstructed_images", str(epoch) + '_target.png'))
					wandb.save(osp.join(self.opt.save_dir, self.opt.model_name,"reconstructed_images", str(epoch) + '_style.png'))
					wandb.save(osp.join(self.opt.save_dir,self.opt.model_name,"reconstructed_images", str(epoch) + '_style_target.png'))
					wandb.save(osp.join(self.opt.save_dir, self.opt.model_name, str(epoch)+"_encoder.pth"))
					wandb.save(osp.join(self.opt.save_dir, self.opt.model_name, str(epoch)+"_decoder.pth"))
					wandb.save(osp.join(self.opt.save_dir,self.opt.model_name, "log_train.txt"))
					

			
			# break
				

	def train_one_epoch(self, epoch):


		self.kl_divergence_losses = AverageMeter()
		self.reconstruction_losses = AverageMeter()
		self.reverse_cycle_losses = AverageMeter()

		# start = time.time()

		for batch_idx in range(int(len(self.trainloader.dataset)/self.opt.batch_size)):

			
			self._forward_cycle()
			self._reverse_cycle()
			self._update_values()
		
			# break
		self._print_values(epoch)


		

	def _forward_cycle(self):
		
		image_batch_1, image_batch_2, _ = next(self.trainloader_iter)
		self.auto_encoder_optimizer.zero_grad()

		self.X_1.copy_(image_batch_1)
		self.X_2.copy_(image_batch_2)

		
		self.style_mu_1, self.style_logvar_1, self.class_latent_space_1 = self.encoder(Variable(self.X_1))
		self.style_latent_space_1 = reparameterize(training=True, mu = self.style_mu_1, logvar = self.style_logvar_1)
	
		# print(self.style_mu_1.size())
		# print(self.style_logvar_1.size())
		# print(self.class_latent_space_1.size())
		
		self.style_mu_2, self.style_logvar_2, self.class_latent_space_2 = self.encoder(Variable(self.X_2))
		self.style_latent_space_2 = reparameterize(training=True, mu = self.style_mu_2, logvar = self.style_logvar_2)

		self.reconstructed_X_1 = self.decoder(self.style_latent_space_1, self.class_latent_space_2)
		self.reconstructed_X_2 = self.decoder(self.style_latent_space_2, self.class_latent_space_1)

		self._compute_forward_loss()
		self.auto_encoder_optimizer.step()




	def _compute_forward_loss(self):


		kl_divergence_loss_1 = self.opt.kl_divergence_coef * (
			- 0.5 * torch.sum(1 + self.style_logvar_1 - self.style_mu_1.pow(2) - self.style_logvar_1.exp())
		)
		kl_divergence_loss_1 /= (self.opt.batch_size * self.opt.num_channels * self.opt.image_size * self.opt.image_size)
		kl_divergence_loss_1.backward(retain_graph=True)


		kl_divergence_loss_2 = self.opt.kl_divergence_coef * (
			- 0.5 * torch.sum(1 + self.style_logvar_2 - self.style_mu_2.pow(2) - self.style_logvar_2.exp())
		)
		kl_divergence_loss_2 /= (self.opt.batch_size * self.opt.num_channels * self.opt.image_size * self.opt.image_size)
		kl_divergence_loss_2.backward(retain_graph=True)


		reconstruction_error_1 = self.opt.reconstruction_coef * mse_loss(self.reconstructed_X_1, Variable(self.X_1))
		reconstruction_error_1.backward(retain_graph=True)

		reconstruction_error_2 = self.opt.reconstruction_coef * mse_loss(self.reconstructed_X_2, Variable(self.X_2))
		reconstruction_error_2.backward()

		self.reconstruction_error = (reconstruction_error_1 + reconstruction_error_2) / self.opt.reconstruction_coef
		self.kl_divergence_error = (kl_divergence_loss_1 + kl_divergence_loss_2) / self.opt.kl_divergence_coef





	def _reverse_cycle(self):
		image_batch_1, _, __ = next(self.trainloader_iter)
		image_batch_2, _, __ = next(self.trainloader_iter)

		self.reverse_cycle_optimizer.zero_grad()

		self.X_1.copy_(image_batch_1)
		self.X_2.copy_(image_batch_2)

		self.style_latent_space.normal_(0., 1.)

		_, __, class_latent_space_1 = self.encoder(Variable(self.X_1))
		_, __, class_latent_space_2 = self.encoder(Variable(self.X_2))

		reconstructed_X_1 = self.decoder(Variable(self.style_latent_space), class_latent_space_1.detach())
		reconstructed_X_2 = self.decoder(Variable(self.style_latent_space), class_latent_space_2.detach())

		style_mu_1, style_logvar_1, _ = self.encoder(reconstructed_X_1)
		style_latent_space_1 = reparameterize(training=False, mu=style_mu_1, logvar=style_logvar_1)

		style_mu_2, style_logvar_2, _ = self.encoder(reconstructed_X_2)
		style_latent_space_2 = reparameterize(training=False, mu=style_mu_2, logvar=style_logvar_2)

		self._compute_reverse_loss(style_latent_space_1,style_latent_space_2)


	def _compute_reverse_loss(self,style_latent_space_1,style_latent_space_2):

		self.reverse_cycle_loss = self.opt.reverse_cycle_coef * l1_loss(style_latent_space_1, style_latent_space_2)
		self.reverse_cycle_loss.backward()
		self.reverse_cycle_loss /= self.opt.reverse_cycle_coef

		self.reverse_cycle_optimizer.step()




	def _update_values(self):
		self.kl_divergence_losses.update(self.kl_divergence_error.data.storage().tolist()[0])
		self.reconstruction_losses.update(self.reconstruction_error.data.storage().tolist()[0])
		self.reverse_cycle_losses.update(self.reverse_cycle_loss.data.storage().tolist()[0])
		
		
		




	def _print_values(self,epoch):

		if(self.opt.debug == False):
			self.experiment.log({'Reconstruction loss': self.reconstruction_losses.mean},step = epoch)
			self.experiment.log({'KL-Divergence loss': self.kl_divergence_losses.mean},step = epoch)
			self.experiment.log({"Reverse cycle loss": self.reverse_cycle_losses.mean},step =  epoch)
			

		print('Reconstruction loss: ' + str(self.reconstruction_error.data.storage().tolist()[0]))
		print('KL-Divergence loss: ' + str(self.kl_divergence_error.data.storage().tolist()[0]))
		print('Reverse cycle loss: ' + str(self.reverse_cycle_loss.data.storage().tolist()[0]))



	def evaluation(self,epoch):
		print("Evaluating Model ...")

		"""
		save reconstructed images and style swapped image generations to check progress
		"""
		image_batch_1, image_batch_2, _ = next(self.valloader_iter)
		image_batch_3, _, __ = next(self.valloader_iter)

		self.X_1.copy_(image_batch_1)
		self.X_2.copy_(image_batch_2)
		self.X_3.copy_(image_batch_3)

		
		style_mu_1, style_logvar_1, _ = self.encoder(Variable(self.X_1))
		_, __, class_latent_space_2 = self.encoder(Variable(self.X_2))
		style_mu_3, style_logvar_3, _ = self.encoder(Variable(self.X_3))

		style_latent_space_1 = reparameterize(training=False, mu=style_mu_1, logvar=style_logvar_1)
		style_latent_space_3 = reparameterize(training=False, mu=style_mu_3, logvar=style_logvar_3)

		reconstructed_X_1_2 = self.decoder(style_latent_space_1, class_latent_space_2)
		reconstructed_X_3_2 = self.decoder(style_latent_space_3, class_latent_space_2)


		# save input image batch
		image_batch = (np.transpose(self.X_1.cpu().numpy(), (0, 2, 3, 1)))
		

		# # image_batch = np.concatenate((image_batch, image_batch, image_batch), axis = 0) * 255
		
		imshow_grid(image_batch, name=str(epoch) + '_original', save=True, opt = self.opt)

		# save reconstructed batch
		reconstructed_x = np.transpose(reconstructed_X_1_2.cpu().data.numpy(), (0, 2, 3, 1))
		# reconstructed_x = np.concatenate((reconstructed_x, reconstructed_x, reconstructed_x), axis=3)
		imshow_grid(reconstructed_x, name=str(epoch) + '_target', save=True, opt = self.opt)

		style_batch = np.transpose(self.X_3.cpu().numpy(), (0, 2, 3, 1))
		# style_batch = np.concatenate((style_batch, style_batch, style_batch), axis=3)
		imshow_grid(style_batch, name=str(epoch) + '_style', save=True, opt = self.opt)

		# # save style swapped reconstructed batch
		reconstructed_style = np.transpose(reconstructed_X_3_2.cpu().data.numpy(), (0, 2, 3, 1))
		# reconstructed_style = np.concatenate((reconstructed_style, reconstructed_style, reconstructed_style), axis=3)
		imshow_grid(reconstructed_style, name=str(epoch) + '_style_target', save=True, opt = self.opt)



	def test(self):

		self.load_model()
		self.evaluation(0)



	def generate_images(self):
		self.load_model()

		print("Generating Images ...")
		next(self.valloader_iter)
		next(self.valloader_iter)
		next(self.valloader_iter)
		next(self.valloader_iter)

		next(self.valloader_iter)
		next(self.valloader_iter)

		
		next(self.valloader_iter)
		next(self.valloader_iter)
		
		


		self.X_1 = torch.FloatTensor(1, self.opt.num_channels, self.opt.image_size, self.opt.image_size)
		self.X_2 = torch.FloatTensor(1, self.opt.num_channels, self.opt.image_size, self.opt.image_size)


		self.X_1 = self.X_1.cuda()
		self.X_2 = self.X_2.cuda()


		total_images = 8


		
		image_batch_1, _ , _ = next(self.valloader_iter)
		image_batch_2, _, __ = next(self.valloader_iter)
		
		image_batch_1 = image_batch_1[:total_images]
		image_batch_2 = image_batch_2[:total_images]


		
		print(image_batch_1.size())
		print(image_batch_2.size())
		
		final_images = []
		style_image_grid = np.zeros((1,60,3))
		for style_count in range(total_images):
			print("*"*40)
			print("CURRENT STYLE: ",style_count)

			current_image_batch_1 = image_batch_1[style_count,:,:,:]
			print(current_image_batch_1.size())
			self.X_1.copy_(current_image_batch_1)



			style_mu_1, style_logvar_1, _ = self.encoder(Variable(self.X_1))
			style_latent_space_1 = reparameterize(training=False, mu=style_mu_1, logvar=style_logvar_1)

			style_output = np.transpose(self.X_1.cpu().numpy(), (0, 2, 3, 1))[0]

			style_output =style_output * 255
			style_output = style_output.astype("uint8")	

			style_image_grid = np.concatenate((style_image_grid, style_output),axis = 0)
		
			print(style_image_grid.shape)
			cv2.imwrite(os.path.join(self.opt.save_dir,"output_images","style_images.png"),style_image_grid)


			# break

			reconstruction_image_grid = np.zeros((60,1,3))
			for target_count in range(total_images):
				print("CURRENT TARGET: ",target_count)
					
				current_image_batch_2 = image_batch_2[target_count,:,:,:]
				
				self.X_2.copy_(current_image_batch_2)

				style_mu_2, style_logvar_2, class_latent_space_2 = self.encoder(Variable(self.X_2))

				style_latent_space_2 = reparameterize(training=False, mu=style_mu_2, logvar=style_logvar_2)

				reconstructed_X_1 = self.decoder(style_latent_space_1, class_latent_space_2)

				reconstructed_image = np.transpose(reconstructed_X_1.cpu().data.numpy(), (0, 2, 3, 1))[0]
				reconstructed_image =reconstructed_image * 255
				reconstructed_image = reconstructed_image.astype("uint8")			

				reconstruction_image_grid = np.concatenate((reconstruction_image_grid,reconstructed_image),axis = 1)

			final_images.append(reconstruction_image_grid)

		# 		# break	
		# 	# break

		
		target_image_grid = np.zeros((60,1,3))
		for target_count in range(total_images):
			print("CURRENT TARGET: ",target_count)
					
			current_image_batch_2 = image_batch_2[target_count,:,:,:]

			print(current_image_batch_2.size())				
			target_output = np.transpose(current_image_batch_2.cpu().data.numpy(), ( 1, 2, 0))
			
			target_output =target_output * 255
			target_output = target_output.astype("uint8")			
					
			target_image_grid = np.concatenate((target_image_grid,target_output), axis = 1)

		cv2.imwrite(os.path.join(self.opt.save_dir,"output_images","target_images.png"),target_image_grid)


			
	

		result_final_image = target_image_grid
		# result_final_image = np.concatenate((result_final_image,final_images[i]),axis=0)

		for i in range(0,len(final_images)):

			result_final_image = np.concatenate((result_final_image,final_images[i]),axis=0)
		
		style_image_grid = style_image_grid[:480,:,:]
		temp = np.zeros((60,60,3))
		style_image_grid = np.concatenate((temp,style_image_grid),axis = 0)
		
		result_final_image = np.concatenate((style_image_grid, result_final_image),axis= 1)
	
		cv2.imwrite(os.path.join(self.opt.save_dir,"output_images","final_grid.png"),result_final_image)
			



	def generate_linear_iterpolation_images(self):
		self.load_model()

		print("Generating Linear Interpolation Images ...")
		
		self.X_1 = torch.FloatTensor(1, self.opt.num_channels, self.opt.image_size, self.opt.image_size)
		self.X_2 = torch.FloatTensor(1, self.opt.num_channels, self.opt.image_size, self.opt.image_size)


		self.X_1 = self.X_1.cuda()
		self.X_2 = self.X_2.cuda()


		images_count = 4000


		
		next(self.trainloader_iter)
		next(self.trainloader_iter)

		next(self.trainloader_iter)
		next(self.trainloader_iter)

		next(self.trainloader_iter)
		next(self.trainloader_iter)

		next(self.trainloader_iter)
		next(self.trainloader_iter)

		next(self.trainloader_iter)
		next(self.trainloader_iter)

		next(self.trainloader_iter)
		next(self.trainloader_iter)

		next(self.trainloader_iter)
		next(self.trainloader_iter)

		next(self.trainloader_iter)
		next(self.trainloader_iter)

		next(self.trainloader_iter)
		next(self.trainloader_iter)

		next(self.trainloader_iter)
		next(self.trainloader_iter)

		next(self.trainloader_iter)
		next(self.trainloader_iter)


		image_batch_1, _ , _ = next(self.trainloader_iter)
		image_batch_2, _ , _ = next(self.trainloader_iter)
		
		image_batch_1 = image_batch_1[:1]
		image_batch_2 = image_batch_2[:1]
		jump = 400
		

		print(image_batch_1.size())
		print(image_batch_2.size())
		

		original_X_1 = np.transpose(image_batch_1.cpu().data.numpy(), (0, 2, 3, 1))[0]
		original_X_2 = np.transpose(image_batch_2.cpu().data.numpy(), (0, 2, 3, 1))[0]


			
		original_X_1 = original_X_1 * 255
		original_X_1 = original_X_1.astype("uint8")			
	

		original_X_2 = original_X_2 * 255
		original_X_2 = original_X_2.astype("uint8")	
		
		cv2.imwrite(os.path.join(self.opt.save_dir,"linear_interpolation_results", "original_x1.png"), original_X_1)
		cv2.imwrite(os.path.join(self.opt.save_dir,"linear_interpolation_results", "original_x2.png"), original_X_2)

		self.X_1.copy_(image_batch_1)
		self.X_2.copy_(image_batch_2)
		
		style_mu_1, style_logvar_1, class_latent_space_1 = self.encoder(Variable(self.X_1))
		style_mu_2, style_logvar_2, class_latent_space_2 = self.encoder(Variable(self.X_2))

		style_latent_space_1 = reparameterize(training=False, mu=style_mu_1, logvar=style_logvar_1)
		style_latent_space_2 = reparameterize(training=False, mu=style_mu_2, logvar=style_logvar_2)

		np_style_latent_space_1 = style_latent_space_1.detach().cpu().numpy()[0]
		np_style_latent_space_2 = style_latent_space_2.detach().cpu().numpy()[0]
		

		np_class_latent_space_1 = class_latent_space_1.detach().cpu().numpy()[0]
		np_class_latent_space_2 = class_latent_space_2.detach().cpu().numpy()[0]
		
		

		z_interpolations = slerpolate(np_style_latent_space_1, np_style_latent_space_2, None, images_count)
		s_interpolations = slerpolate(np_class_latent_space_1,np_class_latent_space_2,None, images_count)
		print(z_interpolations.shape)
		print(s_interpolations.shape)

		result_list = []
		for current_z_count in range(0,images_count,jump):
			
			print("CURRENT Z: ",current_z_count)
			print("*"*40)
			
			current_z = z_interpolations[:,current_z_count]
			
			current_z_tensor = torch.tensor(current_z)
			current_z_tensor = current_z_tensor.view(1,current_z_tensor.size(0))			
			current_z_tensor = current_z_tensor.cuda()

			os.makedirs(os.path.join(self.opt.save_dir,"linear_interpolation_results", str(current_z_count)), exist_ok = True)
			result_image = np.zeros((60,1,3))
			count = 0

			for current_s_count in range(0,images_count,jump):
				print("CURRENT S: ",current_s_count)
	
				current_s = s_interpolations[:,current_s_count]
				
				current_s_tensor = torch.tensor(current_s)
				current_s_tensor = current_s_tensor.view(1,current_s_tensor.size(0))			
				current_s_tensor = current_s_tensor.cuda()

		
				reconstructed_X_1 = self.decoder(current_z_tensor, current_s_tensor)


				output_X_1 = np.transpose(reconstructed_X_1.cpu().data.numpy(), (0, 2, 3, 1))[0]			
				output_X_1 = output_X_1 * 255
				output_X_1 = output_X_1.astype("uint8")			

				# print(output_X_1.shape)
				cv2.imwrite(os.path.join(self.opt.save_dir,"linear_interpolation_results",str(current_z_count),str(current_s_count)+"_x1.png"), output_X_1)
				

				result_image = np.concatenate((result_image, output_X_1), axis=1)

			cv2.imwrite(os.path.join(self.opt.save_dir,"linear_interpolation_results",str(current_z_count),"result.png"), result_image)
			result_list.append(result_image)

			# break
		# print(len(result_list))
		result_final_image = result_list[0]
		for i in range(1,len(result_list)):

			result_final_image = np.concatenate((result_final_image,result_list[i]),axis=0)

		cv2.imwrite(os.path.join(self.opt.save_dir,"linear_interpolation_results","final_result.png"), result_final_image)
		


	def save_model(self,epoch):
		print("SAVING MODEL ..",)
		torch.save(self.encoder.state_dict(), os.path.join(self.opt.save_dir, self.opt.model_name,str(epoch) + "_encoder.pth"))
		torch.save(self.decoder.state_dict(), os.path.join(self.opt.save_dir, self.opt.model_name,str(epoch) +  "_decoder.pth"))





	def load_model(self):
		print("LAODING SAVE MODEL: ",os.path.join(self.opt.save_dir, self.opt.model_name))
		self.encoder.load_state_dict(torch.load(os.path.join(self.opt.save_dir, self.opt.model_name, "99_encoder.pth")))
		self.decoder.load_state_dict(torch.load(os.path.join(self.opt.save_dir, self.opt.model_name, "99_decoder.pth")))
		
		self.encoder.cuda()
		self.decoder.cuda()
