# Imported Libraries
from torch import nn
import lightning as L
from torch.optim import Adam
from .AureliusUnet import AureliusUnet

# LightningModule
class AureliusUnetLightning(L.LightningModule):
	def __init__(self, 
			  	lossFunction:nn.Module,
				learning_rate=1e-3,
				upsample_mode:str="bilinear"):
		super().__init__()
		self.__model = AureliusUnet(upsample_mode=upsample_mode)
		self.__lossFunction = lossFunction
		self.__learning_rate = learning_rate

	def forward(self, x):
		x = self.__model(x)
		return x
	
	def configure_optimizers(self):
		opt = Adam(self.__model.parameters(), lr=self.__learning_rate)
		return opt
	
	def training_step(self, batch, batch_idx):
		x, y = batch
		preds = self.__model(x)
		loss = self.__lossFunction(preds, y)
		self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
		return loss
	
	def validation_step(self, batch, batch_idx):
		x, y = batch
		preds = self.__model(x)
		loss = self.__lossFunction(preds, y)
		self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
		return loss