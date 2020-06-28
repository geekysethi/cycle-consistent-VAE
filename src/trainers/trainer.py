# import os
# import shutil
# import sys
# import time
# from itertools import cycle
# from os import path as osp
# from pprint import pprint

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import wandb
# from torch.autograd import Variable

# from utils.helpers import load_dataloaders, load_model
# from utils.logger import Logger
# from utils.meters import AverageMeter
# from itertools import cycle
# from utils.loss import reparameterize


from trainers.train_classifier import Train_classifier
	


class Trainer:
	def __init__(self, opt):
		
		self.opt = opt		
		torch.manual_seed(opt.seed)
		os.makedirs(opt.save_dir, exist_ok=True)
		sys.stdout = Logger(osp.join(opt.save_dir, opt.model_name,'log_train.txt'))

		print('=========user config==========')
		pprint(opt._state_dict())
		print('============end===============')

		self.trainloader, self.valloader = load_dataloaders(self.opt)
		self.trainloader_iter = cycle(self.trainloader)
		self.valloader_iter = cycle(self.valloader)
		self.use_gpu = opt.use_gpu
		self.device = torch.device('cuda')





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

		self.encoder, self.classifier = load_model(self.opt)
		print("LEARNING RATE: ",self.opt.base_learning_rate)

		self.X_1 = torch.FloatTensor(self.opt.batch_size, self.opt.num_channels, self.opt.image_size, self.opt.image_size)
		self.X_2 = torch.FloatTensor(self.opt.batch_size, self.opt.num_channels, self.opt.image_size, self.opt.image_size)
		self.X_3 = torch.FloatTensor(self.opt.batch_size, self.opt.num_channels, self.opt.image_size, self.opt.image_size)

		self.style_latent_space = torch.FloatTensor(self.opt.batch_size, self.opt.style_dim)


		if self.use_gpu:
			self.device = torch.device('cuda')

			self.encoder.cuda()
			self.classifier.cuda()

			self.X_1 = self.X_1.cuda()
			self.X_2 = self.X_2.cuda()
			self.X_3 = self.X_3.cuda()

			self.style_latent_space = self.style_latent_space.cuda()


		self.load_encoder()


	def _init_optimizer(self):

		"""
		optimizer and scheduler definition
		"""
		self.optimizer = optim.Adam(self.classifier.parameters(),
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

			if(self.opt.debug == False):
				print("UPLOADING FINAL FILES ...")

				wandb.save(osp.join(self.opt.save_dir, self.opt.model_name, "log_train.txt"))
				wandb.save(osp.join(self.opt.save_dir, self.opt.model_name, "model.pth"))
				wandb.save(osp.join(self.opt.save_dir, self.opt.model_name, "best_model.pth"))

				
				# break
	





	def train_one_epoch(self, epoch):

		self.encoder.eval()
		self.classifier.train()
		self.cross_entropy_losses = AverageMeter()
		self.accuracy = AverageMeter()
		correct = 0
		total = 0
	
		for batch_idx in range(int(len(self.trainloader.dataset)/self.opt.batch_size)):
			
			image_batch_1, image_batch_2, labels = next(self.trainloader_iter)
			self.auto_encoder_optimizer.zero_grad()

			self.X_1.copy_(image_batch_1)
			
			
			self.style_mu_1, self.style_logvar_1, self.class_latent_space_1 = self.encoder(Variable(self.X_1))
			style_latent_space_1 = reparameterize(training=True, mu = self.style_mu_1, logvar = self.style_logvar_1)

			outputs = self.classifier(style_latent_space_1)
			
			self.loss = self.cross_entropy_loss(outputs,labels)
			
			loss.backward()
			self.optimizer.step()


			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()
			self.cross_entropy_loss_losses.update(self.loss.item())
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

		self.encoder.eval()
		self.classifier.eval()

		
		self.accuracy = AverageMeter()
		correct = 0
		total = 0
	
		for batch_idx in range(int(len(self.valloader.dataset)/self.opt.batch_size)):
			
			image_batch_1, image_batch_2, labels = next(self.valloader_iter)
			self.auto_encoder_optimizer.zero_grad()

			self.X_1.copy_(image_batch_1)
			
			
			self.style_mu_1, self.style_logvar_1, self.class_latent_space_1 = self.encoder(Variable(self.X_1))
			style_latent_space_1 = reparameterize(training=True, mu = self.style_mu_1, logvar = self.style_logvar_1)

			outputs = self.classifier(style_latent_space_1)
			
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()



		self.accuracy.update(correct/total)			

		if(self.opt.debug == False):
			self.experiment.log({'Validation Accuracy': self.accuracy.mean},step = epoch)


		print('Validation Accuracy: ' + str(self.accuracy.mean))
		return self.accuracy.mean



	

	def save_model(self):
		print("SAVING MODEL ...")
		torch.save(self.encoder.state_dict(), os.path.join(self.opt.save_dir, self.opt.model_name, "encoder.pth"))
		torch.save(self.decoder.state_dict(), os.path.join(self.opt.save_dir, self.opt.model_name, "decoder.pth"))




	def load_encoder(self):

		print("[*] Loading model from {}".format(self.opt.save_dir))
		self.encoder.load_state_dict(torch.load(os.path.join(self.opt.save_dir, self.opt.model_name, "encoder.pth")))

		self.encoder.cuda()






		# if(best_model):
		# 	print("LOADING BEST MODEL")

		# 	filename = "best_model.pth"

		# else:
		# 	filename = "model.pth"

		# ckpt_path = os.path.join(self.opt.save_dir, self.opt.model_name, filename)
		# print(ckpt_path)
		
		# if(self.use_gpu==False):
		# 	self.model=torch.load(ckpt_path, map_location=lambda storage, loc: storage)

		# else:
		# 	print("*"*40+" LOADING MODEL FROM GPU "+"*"*40)
		# 	self.ckpt = torch.load(ckpt_path)
		# 	self.model.load_state_dict(self.ckpt['model_state'])
		# 	self.model.cuda()




