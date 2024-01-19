# Imported Libraries
from torch import nn
import lightning as L
from torch.optim import Adam
from .Unet import Unet

# LightningModule
class UnetLightning(L.LightningModule):
	def __init__(self, 
			  	lossFunction:nn.Module,
				learning_rate=1e-3,
				n_channels=1,
				test_metrics:dict[str, nn.Module] | None = None):
		super().__init__()
		self.__unet = Unet(n_channels=n_channels)
		self.__lossFunction = lossFunction
		self.__learning_rate = learning_rate
		if test_metrics == None:
			self.__test_metrics = {}
		else:
			self.__test_metrics = test_metrics

	def forward(self, x):
		x = self.__unet(x)
		return x
	
	def configure_optimizers(self):
		opt = Adam(self.__unet.parameters(), lr=self.__learning_rate)
		return opt
	
	def training_step(self, batch, batch_idx):
		x, y = batch
		preds = self.__unet(x)
		loss = self.__lossFunction(preds, y)
		self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
		return loss
	
	def validation_step(self, batch, batch_idx):
		x, y = batch
		preds = self.__unet(x)
		loss = self.__lossFunction(preds, y)
		self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
		return loss

	def test_step(self, batch, batch_idx):
		x, y = batch
		preds = self.__unet(x)
		metrics = {}
		for key, loss in self.__test_metrics.items():
			metrics[key] = loss(preds, y)
		metrics["loss"] = self.__lossFunction(preds, y)

		self.log_dict(metrics)
		return metrics