from trainers.train_vae import Train_vae
from config import opt

from utils.helpers import prepare_dirs

def main(opt):
	
	prepare_dirs(opt)
	trainer = Train_vae(opt)

	# trainer.train()
	# # # trainer.model_eval()
	trainer.generate_images()
	trainer.generate_linear_iterpolation_images()

	# trainer.test()

if __name__ == "__main__":
	main(opt)

