# Imported Libraries
from torch import nn
import lightning as L
from torch.optim import Adam
from .DPNUnet import DPNUnet

# LightningModule
class DPNUnetLightning(L.LightningModule):
	def __init__(self, 
			  	lossFunction:nn.Module,
				learning_rate=1e-3,
				upsample_mode:str="nearest"):
		super().__init__()
		self.__dpnunet = DPNUnet(upsample_mode=upsample_mode)
		self.__lossFunction = lossFunction
		self.__learning_rate = learning_rate

	def forward(self, x):
		x = self.__dpnunet(x)
		return x
	
	def configure_optimizers(self):
		opt = Adam(self.__dpnunet.parameters(), lr=self.__learning_rate)
		return opt
	
	def training_step(self, batch, batch_idx):
		x, y, _ = batch
		preds = self.__dpnunet(x)
		loss = self.__lossFunction(preds, y)
		self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
		return loss
	
	def validation_step(self, batch, batch_idx):
		x, y, _ = batch
		preds = self.__dpnunet(x)
		loss = self.__lossFunction(preds, y)
		self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
		return loss