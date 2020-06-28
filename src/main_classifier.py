from trainers.train_classifier import Train_classifier
from config_classifier import opt

from utils.helpers import prepare_dirs

def main(opt):
	
	# prepare_dirs(opt)
	trainer = Train_classifier(opt)

	# trainer.train()
	trainer.visualization()
# 	# # trainer.model_eval()



if __name__ == "__main__":
	main(opt)

