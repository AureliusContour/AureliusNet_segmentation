# Library dependencies
from torch import nn
import torch.utils.checkpoint as checkpoint
from .components import UpLayer
from .components import DownLayer
from .components import ConvBNReLU, DoubleConv

# DPN-Unet architecture
class AureliusUnet(nn.Module):
	def __init__(self, upsample_mode:str="bilinear"):
		"""
		A Full DPNUnet architecture module that takes in a 3x512x512 tensor,
		and outputs a 1x512x512 tensor.

		Arguments it takes are:
		1. upsample_mode (str, optional): Upsampling mode. Default is 'nearest'
		"""
		super(AureliusUnet, self).__init__()
		self.input_conv = DoubleConv(3, 64)

		self.down_layer1 = DownLayer(64, 96, 128, 128, 80, 64, 2)
		self.down_layer2 = DownLayer(128, 192, 256, 256, 160, 128, 3, True)
		self.down_layer3 = DownLayer(288, 320, 512, 512, 288, 256, 6)
		self.down_layer4 = DownLayer(512, 320, 512, 512, 320, 256, 3, True)
		# self.center_block = ConvBNReLU(832, 832)

		self.up_layer4 = UpLayer(1024, 288, upsample_mode=upsample_mode)
		self.up_layer3 = UpLayer(576, 128, upsample_mode=upsample_mode)
		self.up_layer2 = UpLayer(256, 64, upsample_mode=upsample_mode)
		self.up_layer1 = UpLayer(128, 64, upsample_mode=upsample_mode)
		
		# self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
		self.final_conv_block = nn.Conv2d(64, 1, kernel_size=1)
		self.sigmoid = nn.Sigmoid()
	
	def custom_checkpoint_call(self, module, doubleInput=False):
		def custom_forward(*inputs):
			if doubleInput:
				inputs = module(inputs[0], inputs[1])
			else:
				inputs = module(inputs[0])
			return inputs
		return custom_forward
	
	def forward(self, x):
		# Initial
		x1 = self.input_conv(x)
		# Down layers
		down_layer1 = self.down_layer1(x1)
		# down_layer2 = self.down_layer2(down_layer1)
		down_layer2 = checkpoint.checkpoint(self.custom_checkpoint_call(self.down_layer2), down_layer1, use_reentrant=False)
		down_layer3 = self.down_layer3(down_layer2)
		# down_layer4 = self.down_layer4(down_layer3)
		down_layer4 = checkpoint.checkpoint(self.custom_checkpoint_call(self.down_layer4), down_layer3, use_reentrant=False)
		# Center block
		# down_layer4 = self.center_block(down_layer4)
		# Up layers (with skip connections)
		# up_layer4 = self.up_layer4(down_layer4, down_layer3)
		x = checkpoint.checkpoint(self.custom_checkpoint_call(self.up_layer4, doubleInput=True), down_layer4, down_layer3, use_reentrant=False)
		x = self.up_layer3(x, down_layer2)
		# up_layer2 = self.up_layer2(up_layer3, down_layer1)
		x = checkpoint.checkpoint(self.custom_checkpoint_call(self.up_layer2, doubleInput=True), x, down_layer1, use_reentrant=False)
		x = self.up_layer1(x, x1)
		# Output
		# x = self.upsample(up_layer1)
		x = self.final_conv_block(x)
		x = self.sigmoid(x)
		return x

