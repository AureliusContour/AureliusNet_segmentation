# PyTorch libraries
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

# Local libraries
from .early_stopping import EarlyStopping

# Other libraries
from tqdm import tqdm
from datetime import datetime
import os

class ModelTrainer():
	def __init__(self, model:torch.nn.Module, 
			  	lossFunction:torch.nn.Module,
				optimizer,
				output_checkpoints_path,
				output_runs_path,
				total_epochs=100,
				early_stopping_patience=None,
				early_stopping_min_delta=0.0,
				device="cpu") -> None:
		# Training 
		self.__model = model.to(device)
		self.__lossFunction = lossFunction
		self.__opt = optimizer
		self.total_epochs = total_epochs
		self.current_epoch = 1
		self.device = device
		# Logging 
		self.output_checkpoints_path = output_checkpoints_path
		self.output_runs_path = output_runs_path
		# Callback 
		if early_stopping_patience == None:
			self.early_stopping = None
		else:
			self.early_stopping = EarlyStopping(
				early_stopping_patience,
				early_stopping_min_delta,
			)
		# Data
		self.trainLoader = None
		self.no_train_batches = None
		self.valLoader = None
		self.no_val_batches = None
	
	def load_checkpoint(self, path):
		"""
		Load previous a training run's checkpoint.
		Takes in arguments:
		1. path (str): path to .pt file
		"""
		checkpoint = torch.load(path)
		self.__model.load_state_dict(checkpoint['model_state_dict'])
		self.__opt.load_state_dict(checkpoint['optimizer_state_dict'])
		self.early_stopping.setBestLoss(checkpoint['loss'])
		self.current_epoch = checkpoint["epoch"]

	def train_per_epoch(self) -> float:
		"""
		Per epoch training processes
		"""
		self.__model.train()
		total_loss = 0
		# loop over the training set
		with tqdm(total=self.no_train_batches) as pbar:
			pbar.set_description(f"Training Batch 1/{self.no_train_batches}")
			for i, (x, y, z) in enumerate(self.trainLoader):
				# Zero gradients per batch
				self.__opt.zero_grad()
				# send the input to the device
				x, y = (x.to(self.device), y.to(self.device))
				# Make a prediction for this batch
				pred = self.__model(x)
				# Compute loss and gradients
				loss = self.__lossFunction(pred, y)
				loss.backward()
				# Adjust learning weights
				self.__opt.step()
				total_loss += loss
				# Update loading bar
				if i + 1 < self.no_train_batches:
					pbar.set_description(f"Training Batch {i+2}/{self.no_train_batches}. Training loss = {(total_loss / (i+1)):.4f}")
				else:
					pbar.set_description(f"Training loss = {(total_loss / (i+1)):.4f}")
				pbar.update(1)

		return total_loss / self.no_train_batches
	
	def validation_per_epoch(self) -> float:
		"""
		Per epoch validation processes
		"""
		total_loss = 0
		with torch.no_grad():
			self.__model.eval()
			# loop over the validation set
			with tqdm(total=self.no_val_batches) as pbar:
				pbar.set_description(f"Validating Batch 1/{self.no_val_batches}.")
				for i, (x, y, z) in enumerate(self.valLoader):
					# send the input to the device
					x, y = (x.to(self.device), y.to(self.device))
					# make the predictions and calculate the validation loss
					pred = self.__model(x)
					total_loss += self.__lossFunction(pred, y)
					if i + 1 < self.no_val_batches:
						pbar.set_description(f"Validating Batch {i+2}/{self.no_val_batches}. Validation loss = {(total_loss / (i+1)):.4f}")
					else:
						pbar.set_description(f"Validation loss = {(total_loss / (i+1)):.4f}")
					pbar.update(1)
		
		return total_loss / self.no_val_batches
	
	def train(self, train_dataset, val_dataset, batch_size=32, pin_memory=False, num_workers=os.cpu_count()):
		self.trainLoader = DataLoader(train_dataset,
								shuffle=True,
								batch_size=batch_size,
								pin_memory=pin_memory,
								num_workers=num_workers
								)
		self.valLoader = DataLoader(val_dataset,
								shuffle=True,
								batch_size=batch_size,
								pin_memory=pin_memory,
								num_workers=num_workers
								)
		self.no_train_batches = len(train_dataset) // batch_size
		self.no_val_batches = len(val_dataset) // batch_size

		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		output_checkpoint_path = os.path.join(self.output_checkpoints_path, f'dpnunet_breast_ct_{timestamp}')
		output_runs_path = os.path.join(self.output_runs_path, f'dpnunet_breast_ct_{timestamp}')
		os.mkdir(output_checkpoint_path)
		writer = SummaryWriter(output_runs_path)
		print(f"Training DPNUnet Model with {self.no_train_batches} train batches and {self.no_val_batches} validation batches")
		print(f"Storing runs' tensorboard loss statistics on '{output_runs_path}'")
		print(f"Storing runs' checkpoints on '{output_checkpoint_path}/'")
		while self.current_epoch <= self.total_epochs:
			print(f"\nEPOCH {self.current_epoch}/{self.total_epochs}")
			avg_train_loss = self.train_per_epoch()
			avg_val_loss = self.validation_per_epoch()

			# Logging
			writer.add_scalars('Loss/train', {"Training": avg_train_loss}, self.current_epoch)
			writer.add_scalars('Loss/valid', {"Validation": avg_val_loss}, self.current_epoch)
			writer.add_scalars('Loss/train_valid', 
				{"Training": avg_train_loss, "Validation": avg_val_loss}, self.current_epoch)
			writer.flush()

			# Save last model's state as checkpoint for resuming
			last_path = os.path.join(output_checkpoint_path, "last.pt")
			torch.save({
				"epoch": self.current_epoch,
				"model_state_dict": self.__model.state_dict(), 
				"optimizer_state_dict": self.__opt.state_dict(),
				"loss": avg_val_loss,
			}, last_path)

			# Track best performance, and save the model's state
			if avg_val_loss < self.early_stopping.best_loss:
				self.early_stopping.setBestLoss(avg_val_loss)
				best_path = os.path.join(output_checkpoint_path, "best.pt")
				torch.save({
					"model_state_dict": self.__model.state_dict(), 
					"loss": avg_val_loss}, 
					best_path
				)
			# Early stopping check
			if self.early_stopping.check(avg_val_loss):
				print(f"Early Stopping Triggered! Training stopped at epoch {self.current_epoch}.")
				break
			self.current_epoch += 1
		self.current_epoch = 0
		return self.__model