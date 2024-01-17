# Library dependencies
from torch import nn, sigmoid
import torch.utils.checkpoint as checkpoint
from .components import ConvBNReLU, DoubleConv, AureliusDownLayer, AureliusUpLayer

# DPN-Unet architecture
class AureliusUnet(nn.Module):
	def __init__(self, in_channels=3, upsample_mode:str="bilinear"):
		"""
		A Full DPNUnet architecture module that takes in a 3x512x512 tensor,
		and outputs a 1x512x512 tensor.

		Arguments it takes are:
		1. upsample_mode (str, optional): Upsampling mode. Default is 'nearest'
		"""
		super(AureliusUnet, self).__init__()
		self.input_conv = DoubleConv(in_channels, 64)
		self.down_layer1 = AureliusDownLayer(64, 48, 48, 128, 16)
		self.down_layer2 = AureliusDownLayer(144, 96, 96, 256, 32, True)
		self.down_layer3 = AureliusDownLayer(288, 192, 192, 512, 64)
		self.down_layer4 = AureliusDownLayer(576, 384, 384, 1024, 128, True)
		self.center_block = ConvBNReLU(1152, 576)
		self.up_layer4 = AureliusUpLayer(1152, 288, True)
		self.up_layer3 = AureliusUpLayer(576, 144)
		self.up_layer2 = AureliusUpLayer(288, 72, True)
		self.up_layer1 = AureliusUpLayer(136, 64)
		self.final_conv_block = DoubleConv(64, 1)
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
		input_x = self.input_conv(x)
		# Down layers
		down_layer1 = self.down_layer1(input_x)
		# down_layer2 = self.down_layer2(down_layer1)
		down_layer2 = checkpoint.checkpoint(self.custom_checkpoint_call(self.down_layer2), down_layer1, use_reentrant=False)
		down_layer3 = self.down_layer3(down_layer2)
		# down_layer4 = self.down_layer4(down_layer3)
		down_layer4 = checkpoint.checkpoint(self.custom_checkpoint_call(self.down_layer4), down_layer3, use_reentrant=False)
		# Center block
		x = self.center_block(down_layer4)
		# Up layers (with skip connections)
		# x = self.up_layer4(x, down_layer3)
		x = checkpoint.checkpoint(self.custom_checkpoint_call(self.up_layer4, doubleInput=True), x, down_layer3, use_reentrant=False)
		x = self.up_layer3(x, down_layer2)
		# x = self.up_layer2(x, down_layer1)
		x = checkpoint.checkpoint(self.custom_checkpoint_call(self.up_layer2, doubleInput=True), x, down_layer1, use_reentrant=False)
		x = self.up_layer1(x, input_x)
		# Output
		x = self.final_conv_block(x)
		x = self.sigmoid(x)
		return x

