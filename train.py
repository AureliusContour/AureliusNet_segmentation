# PyTorch Lightning libraries
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import RichModelSummary, ModelCheckpoint, EarlyStopping, RichProgressBar, BatchSizeFinder, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

# Local libraries
from datasets import BreastCTDataset, transform, BreastCTDataModule
from models import DPNUnetLightning
from utils.losses_and_metrics import DiceLoss

# Other libraries
import pandas as pd
import os
import wandb
from argparse import ArgumentParser, Namespace
from datetime import datetime
import yaml


def preprocess(data: pd.DataFrame, config:dict) -> (BreastCTDataset, BreastCTDataset):
	# convert to Dataset
	trainDS = BreastCTDataset(data, transform=transform, set_type="train")
	valDS = BreastCTDataset(data, transform=transform, set_type="validation")
	print(f"[INFO] found {len(trainDS)} examples in the training set...")
	print(f"[INFO] found {len(valDS)} examples in the validation set...")
	trainLoader = DataLoader(trainDS, 
						  batch_size=config["training"]["batch_size"], 
						  shuffle=True, 
						  num_workers=os.cpu_count(), 
						  pin_memory=True, 
						  persistent_workers=True)
	valLoader = DataLoader(valDS, 
						batch_size=config["training"]["batch_size"], 
						shuffle=False, 
						num_workers=os.cpu_count(), 
						pin_memory=True, 
						persistent_workers=True)
	return trainLoader, valLoader

def default_management(parsed_args: Namespace, config_dict: dict) -> None:
	"""
	Check for arguments in config and parsed args, and decides which value to take.
	If both are empty, then a default value is assigned.
	"""
	if parsed_args.device != None:
		config_dict["device"] = parsed_args.device
	elif config_dict.get("device") == None:
		config_dict["device"] = "auto" # Default if both config and arg is empty

	if parsed_args.learningrate != None:
		config_dict["training"]["learning_rate"] = parsed_args.learningrate
	elif config_dict.get("training", {}).get("learning_rate") == None:
		config_dict["training"]["learning_rate"] = 0.001 # Default if both config and arg is empty

	if parsed_args.epoch != None:
		config_dict["training"]["num_epochs"] = parsed_args.epoch
	elif config_dict.get("training", {}).get("num_epochs") == None:
		config_dict["training"]["num_epochs"] = 100 # Default if both config and arg is empty

	if parsed_args.batchsize != None:
		config_dict["training"]["batch_size"] = parsed_args.batchsize
	elif config_dict.get("training", {}).get("batch_size") == None:
		config_dict["training"]["batch_size"] = 32 # Default if both config and arg is empty
	
	if parsed_args.patience != None:
		config_dict["training"]["early_stopping_patience"] = parsed_args.patience
	elif config_dict.get("training", {}).get("early_stopping_patience") == None:
		config_dict["training"]["early_stopping_patience"] = config_dict["training"]["num_epochs"] // 10 # Default if both config and arg is empty
	
def main():
	# Time of execution
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

	# Arguments parsing
	parser = ArgumentParser(
                    prog='DPNUnet Trainer',
                    description='Trains the DPNUnet model on a specified dataset',
                    epilog='Make sure you include dataset csv path and the config yaml path!')
	
	parser.add_argument("-d", "--dataset", metavar="Dataset CSV", help="Path to dataset csv.", required=True)
	parser.add_argument("-c", "--config", metavar="Config YAML", help="Path to config yaml", required=True)
	parser.add_argument("-p", "--project", metavar="Project Name", help="wandb project name", default="BreastCancerCT_DPNUnet")
	parser.add_argument("-n", "--name", metavar="Run Name", help="wandb project run name", default=timestamp)
	parser.add_argument("-r", "--resume", metavar="Resume Checkpoint", help="Path to .ckpt file", default=None)
	parser.add_argument("-de", "--device", metavar="Accelerator device type", choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto"])
	parser.add_argument("-lr", "--learningrate", type=float, metavar="Optimizer Learning Rate", help="Optimally between 0.0001 and 0.01 for most cases")
	parser.add_argument("-ep", "--epoch", type=int, metavar="Number of Epoch")
	parser.add_argument("-bs", "--batchsize", type=int, metavar="Batch Size", help="ideally 32, but can adjust according to needs...")
	parser.add_argument("-pa", "--patience", type=int, metavar="Early Stopping Patience", help="ideally would be 10% of the total number of epochs.")
	parser.add_argument("-fd", "--fastdevrun", action="store_true", help="Developer Run only takes in one batch and one epoch")
	parser.add_argument("-bsf", "--batchsizefinder", action="store_true", help="Experimental feature (MIGHT NOT WORK!)")

	# Load args, dataset and config
	ARGS = parser.parse_args()
	DATASET = pd.read_csv(ARGS.dataset)
	with open(ARGS.config, "r") as c:
		CONFIG = yaml.safe_load(c)
	print(f"Training {ARGS.name}!\n")

	# Do default params management
	default_management(ARGS, CONFIG)

	# Preprocess dataloader
	data_module = BreastCTDataModule(DATASET, CONFIG["training"]["batch_size"])
	print(f"[INFO] found {len(DATASET[DATASET['set'] == 'train'])} examples in the training set...")
	print(f"[INFO] found {len(DATASET[DATASET['set'] == 'validation'])} examples in the validation set...")
	# trainLoader, valLoader = preprocess(DATASET, CONFIG)

	# Initialize DPNUnet Model
	dice_loss = DiceLoss()
	dpnunet = DPNUnetLightning(
				lossFunction=dice_loss,
				learning_rate=CONFIG["training"]["learning_rate"]
			)
	print(f'\nDPNUnet Model initialized and optimized with a learning rate of {CONFIG["training"]["learning_rate"]}')

	# Initialize callbacks
	model_summary_callback = RichModelSummary(max_depth=3, theme="")
	early_stopping_callback = EarlyStopping("val_loss", patience=CONFIG["training"]["early_stopping_patience"], mode="min")
	pbar = RichProgressBar(leave=True, refresh_rate=1)
	tbar = TQDMProgressBar()
	# checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
	wandb_logger = WandbLogger(project=ARGS.project, log_model="all", name=ARGS.name, checkpoint_name=f"model_{ARGS.project}_{ARGS.name}")
	callbacks = [model_summary_callback, early_stopping_callback, tbar]
	
	if ARGS.batchsizefinder:
		bsf = BatchSizeFinder("binsearch", init_val=4)
		callbacks.append(bsf)
	
	# Trainer
	trainer = Trainer(callbacks=callbacks,
				  logger=wandb_logger,
				  accelerator=CONFIG["device"],
				log_every_n_steps=1,
				max_epochs=CONFIG["training"]["num_epochs"],
				min_epochs=1,
				enable_checkpointing=False,
				fast_dev_run=ARGS.fastdevrun)
	print() #empty line

	if ARGS.resume != None:
		trainer.fit(dpnunet, datamodule=data_module, ckpt_path=ARGS.resume)
	else:
		trainer.fit(dpnunet, datamodule=data_module)
	

if __name__ == "__main__":
	main()