from trainers.train_prediction import Train_prediction
from config_prediction import opt

from utils.helpers import prepare_dirs

def main(opt):
	
	prepare_dirs(opt)
	trainer = Train_prediction(opt)
	trainer.evaluation(99)
	# trainer.train()
# 	# # trainer.model_eval()



if __name__ == "__main__":
	main(opt)

