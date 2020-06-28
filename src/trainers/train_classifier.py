import os
import shutil
import sys
import time
from itertools import cycle
from os import path as osp
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.autograd import Variable

from utils.helpers import load_dataloaders, load_model
from utils.logger import Logger
from utils.loss import reparameterize
from utils.meters import AverageMeter

from matplotlib import pyplot as plt
import cv2

class Train_classifier:
	def __init__(self, opt):
		
		self.opt = opt		
		torch.manual_seed(opt.seed)


		print('=========user config==========')
		pprint(opt._state_dict())
		print('============end===============')

		self.trainloader, self.valloader = load_dataloaders(self.opt)
		
		self.use_gpu = opt.use_gpu
		self.device = torch.device('cuda')





		self._init_model()
		self._init_criterion()
		self._init_optimizer()

		self.model_dir = os.path.join(self.opt.save_dir,str(self.opt.model_type) + "_"+str(self.opt.model_name))

		Path(self.model_dir).mkdir(parents=True, exist_ok=True)

		if not os.path.exists(os.path.join(self.model_dir, 'reconstructed_images')):
			os.makedirs(os.path.join(self.model_dir, 'reconstructed_images'))
			

		
		if(self.opt.debug == False):
			self.experiment = wandb.init(project="cycle_consistent_vae")
			hyper_params = self.opt._state_dict()
			self.experiment.config.update(hyper_params)
			wandb.watch(self.encoder)
			wandb.watch(self.classifier_model)



	def _init_model(self):


		self.encoder, self.classifier_model = load_model(self.opt)


		print("LEARNING RATE: ",self.opt.base_learning_rate)

		self.X_1 = torch.FloatTensor(self.opt.batch_size, self.opt.num_channels, self.opt.image_size, self.opt.image_size)
		self.X_2 = torch.FloatTensor(self.opt.batch_size, self.opt.num_channels, self.opt.image_size, self.opt.image_size)
		self.X_3 = torch.FloatTensor(self.opt.batch_size, self.opt.num_channels, self.opt.image_size, self.opt.image_size)

		self.style_latent_space = torch.FloatTensor(self.opt.batch_size, self.opt.style_dim)


		if self.use_gpu:
			self.device = torch.device('cuda')

			self.encoder.cuda()
			self.classifier_model.cuda()


			self.X_1 = self.X_1.cuda()
			self.X_2 = self.X_2.cuda()
			self.X_3 = self.X_3.cuda()

			self.style_latent_space = self.style_latent_space.cuda()


		self.load_encoder()



	def _init_optimizer(self):

		"""
		optimizer and scheduler definition
		"""
		self.optimizer = optim.Adam(self.classifier_model.parameters(),
									lr=self.opt.base_learning_rate,
									betas = (self.opt.beta1, self.opt.beta2))
		# divide the learning rate by a factor of 10 after 80 epochs
		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=80, gamma=0.1)



	def _init_criterion(self):

		"""
		loss definitions
		"""
		self.cross_entropy_loss = nn.CrossEntropyLoss()
		self.cross_entropy_loss.cuda()




	def train(self):
		
			sys.stdout = Logger(osp.join(self.model_dir,'log_train.txt'))

			print("TRAINING CLASSIFIER...")
		
			start_epoch = self.opt.start_epoch
			best_acc = 0.0
			best_epoch = 0

			for epoch in range(start_epoch, self.opt.total_epochs):
		
				print('')
				print('Epoch #' + str(epoch) + '..........................................................................')
				
				self.train_one_epoch(epoch)
				val_acc = self.evaluation(epoch)

				self.scheduler.step()

				if(val_acc > best_acc):
					best_acc = val_acc

					self.save_model(best_model = True)
				

				# break
	
			self.save_model(best_model = False)
			if(self.opt.debug == False):

				print("UPLOADING FINAL FILES ...")

				wandb.save(self.model_dir + "/*")
				# wandb.save(os.path.join(self.opt.save_dir, self.opt.model_name, 
				# 			str(self.opt.model_type)+ "_classifier.pth"))
				
				# wandb.save(os.path.join(self.opt.save_dir, self.opt.model_name, 
				# 			str(self.opt.model_type)+ "_best_classifier.pth"))

				
			





	def train_one_epoch(self, epoch):

		# self.encoder.eval()
		self.classifier_model.train()

		self.cross_entropy_losses = AverageMeter()
		self.accuracy = AverageMeter()
		correct = 0
		total = 0
	
		for batch_idx, data in enumerate(self.trainloader) :
			
			image_batch_1, image_batch_2, labels = data
			labels = labels.cuda()
			# labels = torch.FloatTensor(labels).cuda()
			self.optimizer.zero_grad()
			self.X_1.copy_(image_batch_1)
			self.style_mu_1, self.style_logvar_1, self.class_latent_space_1 = self.encoder(Variable(self.X_1))
			style_latent_space_1 = reparameterize(training = False, mu = self.style_mu_1, logvar = self.style_logvar_1)
			
			if(self.opt.model_type == "specified"):

				outputs = self.classifier_model(style_latent_space_1)

			else:

				outputs = self.classifier_model(self.class_latent_space_1)
			
			self.loss = self.cross_entropy_loss(outputs,labels)
			
			
			self.loss.backward()
			self.optimizer.step()


			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()
			self.cross_entropy_losses.update(self.loss.item())

			del(image_batch_1)
			del(image_batch_2)
			del(outputs)
			del(labels)

			# break

		self.accuracy.update(correct/total)			
		self._print_values(epoch)


		
	def _print_values(self,epoch):

		if(self.opt.debug == False):
			self.experiment.log({'Cross Entropy loss': self.cross_entropy_losses.mean},step = epoch)
			self.experiment.log({'Train Accuracy': self.accuracy.mean},step = epoch)

		print('Cross Entropy loss: ' + str(self.cross_entropy_losses.mean))
		print('Train Accuracy: ' + str(self.accuracy.mean))


	def evaluation(self,epoch):
		print("Evaluating Model ...")

		# self.encoder.eval()
		self.classifier_model.eval()


		
		self.val_accuracy = AverageMeter()
		correct = 0
		total = 0
		with torch.no_grad():

			for batch_idx, data in enumerate(self.valloader) :
				
				image_batch_1, image_batch_2, labels = data

				self.X_1.copy_(image_batch_1)
				
				labels = labels.cuda()
				
				self.style_mu_1, self.style_logvar_1, self.class_latent_space_1 = self.encoder(Variable(self.X_1))
				style_latent_space_1 = reparameterize(training = False, mu = self.style_mu_1, logvar = self.style_logvar_1)


				if(self.opt.model_type == "specified"):
					outputs = self.classifier_model(style_latent_space_1)


				else:
					outputs = self.classifier_model(self.class_latent_space_1)


				_, predicted = outputs.max(1)
				total += labels.size(0)
				correct += predicted.eq(labels).sum().item()

				# break

		self.val_accuracy.update(correct/total)			

		if(self.opt.debug == False):
			self.experiment.log({'Validation Accuracy': self.val_accuracy.mean},step = epoch)


		print('Validation Accuracy: ' + str(self.val_accuracy.mean))
		print("Correct: ",correct)
		print("Total: ",total)
		return self.val_accuracy.mean



	def visualization(self):
		
		self.load_classifier(True)

		self.classifier_model.eval()
		val_iter = iter(self.valloader)
		data = val_iter.next()

		with torch.no_grad():
			image_batch_1, image_batch_2, labels = data

			self.X_1.copy_(image_batch_1)
				
			labels = labels.cuda()
			
			self.style_mu_1, self.style_logvar_1, self.class_latent_space_1 = self.encoder(Variable(self.X_1))
			style_latent_space_1 = reparameterize(training = False, mu = self.style_mu_1, logvar = self.style_logvar_1)


			if(self.opt.model_type == "specified"):

				outputs = self.classifier_model(style_latent_space_1)

			else:
				outputs = self.classifier_model(self.class_latent_space_1)



				
			_, predicted = outputs.max(1)
			image_batch = (np.transpose(self.X_1.cpu().numpy(), (0, 2, 3, 1)))
		
			labels = labels.detach().cpu().numpy()

			predicted = predicted.detach().cpu().numpy()
			

		shape=[2, 8]
		fig = plt.figure(1)
		grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)
		size = shape[0] * shape[1]





		for i in range(size):
			current_image = image_batch[i]

			current_image = current_image * 255
			current_image = current_image.astype("uint8")

			current_image = cv2.UMat(current_image).get()
			
			if(labels[i]== predicted[i]):

				cv2.rectangle(current_image,(0,0),(60,60),(0,255,0),2)

			else:

				cv2.rectangle(current_image,((0),0),(60,60),(255,0,0),2)

			grid[i].axis('off')
			grid[i].imshow(current_image)  # The AxesGrid object work as a list of axes.

		print("SAVING IMAGES")
		path = os.path.join(self.model_dir,'missclassification.png')
		plt.savefig(path)
		plt.clf()




	def save_model(self,best_model):
		print("SAVING MODEL ...")

		if(best_model):
			torch.save(self.classifier_model.state_dict(), os.path.join(self.model_dir, "best_classifier.pth"))

		else:
			torch.save(self.classifier_model.state_dict(),os.path.join(self.model_dir, "classifier.pth"))



	def load_encoder(self):

		print("[*] LOADING ENCODER: {}".format(os.path.join(self.opt.save_dir, "vae", "encoder.pth")))
		self.encoder.load_state_dict(torch.load(os.path.join(self.opt.save_dir, "vae", "encoder.pth")))

		self.encoder.cuda()


	def load_decoder(self):

		print("[*] LOADING DECODER: {}".format(os.path.join(self.opt.save_dir, "vae", "decoder.pth")))
		self.decoder.load_state_dict(torch.load(os.path.join(self.opt.save_dir, "vae", "decoder.pth")))
		self.decoder.cuda()



	def load_classifier(self,best_model):

		if(best_model):
			print("LOADING BEST MODEL")

			self.classifier_model.load_state_dict(torch.load(os.path.join(self.model_dir, "best_classifier.pth")))

		else:

			self.classifier_model.load_state_dict(torch.load(os.path.join(self.model_dir, "_classifier.pth")))


		self.classifier_model.cuda()