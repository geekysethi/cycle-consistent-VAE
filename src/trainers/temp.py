	def generate_images(self):
		self.load_model()
		# self.encoder.eval()
		# self.decoder.eval()
		
		self.X_1 = torch.FloatTensor(1, self.opt.num_channels, self.opt.image_size, self.opt.image_size)
		self.X_2 = torch.FloatTensor(1, self.opt.num_channels, self.opt.image_size, self.opt.image_size)
		self.X_3 = torch.FloatTensor(1, self.opt.num_channels, self.opt.image_size, self.opt.image_size)


		self.X_1 = self.X_1.cuda()
		self.X_2 = self.X_2.cuda()
		self.X_3 = self.X_3.cuda()


		total_images = 8


		print("Generating Images ...")
		next(self.valloader_iter)
		next(self.valloader_iter)
		next(self.valloader_iter)
		next(self.valloader_iter)

		next(self.valloader_iter)
		next(self.valloader_iter)
		next(self.valloader_iter)
		next(self.valloader_iter)

		
		next(self.valloader_iter)
		next(self.valloader_iter)
		
		next(self.valloader_iter)
		next(self.valloader_iter)
		

		# next(self.valloader_iter)
		next(self.valloader_iter)
		
		image_batch_1, image_batch_2, _ = next(self.valloader_iter)
		image_batch_3, _, __ = next(self.valloader_iter)
		
		image_batch_1 = image_batch_1[:total_images]
		image_batch_2 = image_batch_2[:total_images]
		image_batch_3 = image_batch_3[:total_images]


		
		print(image_batch_1.size())
		print(image_batch_3.size())
		
		final_images = []
		for style_count in range(total_images):
			print("*"*40)
			print("CURRENT STYLE: ",style_count)

			current_image_batch_3 = image_batch_3[style_count,:,:,:]

			print(current_image_batch_3.size())
			
			self.X_3.copy_(current_image_batch_3)



			style_mu_3, style_logvar_3, _ = self.encoder(Variable(self.X_3))
			style_output = np.transpose(self.X_3.cpu().numpy(), (0, 2, 3, 1))[0]

			style_output =style_output * 255
			style_output = style_output.astype("uint8")			

			style_name = "{}".format(str(style_count).zfill(8))
			os.makedirs(os.path.join(self.opt.save_dir,"output_images",str(style_name)), exist_ok=True)
			

			cv2.imwrite(os.path.join(self.opt.save_dir,"output_images",str(style_name),"style.png"),style_output)
			
			# break
			reconstruction_image_grid = np.zeros((60,1,3))
			for target_count in range(total_images):
				print("CURRENT TARGET: ",target_count)
					
				current_image_batch_1 = image_batch_1[target_count,:,:,:]
				current_image_batch_2 = image_batch_2[target_count,:,:,:]
				
				
				self.X_1.copy_(current_image_batch_1)
				self.X_2.copy_(current_image_batch_2)

				style_mu_1, style_logvar_1, _ = self.encoder(Variable(self.X_1))
				_, __, class_latent_space_2 = self.encoder(Variable(self.X_2))

				style_latent_space_1 = reparameterize(training=False, mu=style_mu_1, logvar=style_logvar_1)
				style_latent_space_3 = reparameterize(training=False, mu=style_mu_3, logvar=style_logvar_3)

				reconstructed_X_1_2 = self.decoder(style_latent_space_1, class_latent_space_2)
				reconstructed_X_3_2 = self.decoder(style_latent_space_3, class_latent_space_2)


				target_output = np.transpose(reconstructed_X_1_2.cpu().data.numpy(), (0, 2, 3, 1))[0]
				reconstructed_image = np.transpose(reconstructed_X_3_2.cpu().data.numpy(), (0, 2, 3, 1))[0]


				

				target_output =target_output * 255
				target_output = target_output.astype("uint8")			
					
				reconstructed_image =reconstructed_image * 255
				reconstructed_image = reconstructed_image.astype("uint8")			



				
				target_name = "{}".format(str(target_count).zfill(8))
			

				cv2.imwrite(os.path.join(self.opt.save_dir,"output_images",str(style_name),str(target_name)+"_target.png"),target_output)
				cv2.imwrite(os.path.join(self.opt.save_dir,"output_images",str(style_name),str(target_name)+"reconstructed.png"),reconstructed_image)
			
				reconstruction_image_grid = np.concatenate((reconstruction_image_grid,reconstructed_image),axis = 1)

			cv2.imwrite(os.path.join(self.opt.save_dir,"output_images",str(style_name),"final_reconstructed.png"),reconstruction_image_grid)
			final_images.append(reconstruction_image_grid)

				# break	
			# break

		result_final_image = final_images[0]
		for i in range(1,len(final_images)):

			result_final_image = np.concatenate((result_final_image,final_images[i]),axis=0)

		cv2.imwrite(os.path.join(self.opt.save_dir,"output_images","final_grid.png"),result_final_image)
