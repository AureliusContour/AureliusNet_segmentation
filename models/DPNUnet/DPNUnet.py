# Library dependencies
from torch import nn
import torch.utils.checkpoint as checkpoint
from .components import UpLayer
from .components import DownLayer
from .components import ConvBNReLU, DoubleConv

# DPN62-Unet architecture
class DPN68Unet(nn.Module):
	def __init__(self, upsample_mode:str="bilinear"):
		"""
		A Full DPNUnet architecture module that takes in a 3x512x512 tensor,
		and outputs a 1x512x512 tensor.

		Arguments it takes are:
		1. upsample_mode (str, optional): Upsampling mode. Default is 'nearest'
		"""
		super(DPN68Unet, self).__init__()
		self.input_conv = DoubleConv(3, 64)

		self.down_layer1 = DownLayer(64, 96, 128, 128, 80, 64, 3)
		self.down_layer2 = DownLayer(144, 192, 256, 256, 160, 128, 4, True)
		self.down_layer3 = DownLayer(320, 320, 512, 512, 288, 256, 12)
		self.down_layer4 = DownLayer(704, 320, 512, 512, 320, 256, 3, True)
		# self.center_block = ConvBNReLU(832, 832)

		self.up_layer4 = UpLayer(1216, 320, upsample_mode=upsample_mode)
		self.up_layer3 = UpLayer(640, 144, upsample_mode=upsample_mode)
		self.up_layer2 = UpLayer(288, 64, upsample_mode=upsample_mode)
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


# DPN92-Unet architecture
class DPN92Unet(nn.Module):
	def __init__(self, upsample_mode:str="bilinear"):
		"""
		A Full DPNUnet architecture module that takes in a 3x512x512 tensor,
		and outputs a 1x512x512 tensor.

		Arguments it takes are:
		1. upsample_mode (str, optional): Upsampling mode. Default is 'nearest'
		"""
		super(DPN92Unet, self).__init__()
		self.input_conv = DoubleConv(3, 64)

		self.down_layer1 = DownLayer(64, 288, 96, 96, 272, 256, 3)
		self.down_layer2 = DownLayer(336, 576, 192, 192, 544, 512, 4, True)
		self.down_layer3 = DownLayer(704, 1072, 384, 384, 1048, 1024, 20)
		self.down_layer4 = DownLayer(1552, 2304, 768, 768, 2176, 2048, 3, True)
		# self.center_block = ConvBNReLU(832, 832)

		self.up_layer4 = UpLayer(4240, 704, upsample_mode=upsample_mode)
		self.up_layer3 = UpLayer(1408, 336, upsample_mode=upsample_mode)
		self.up_layer2 = UpLayer(672, 64, upsample_mode=upsample_mode)
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

class PaperDPNUnet(nn.Module):
	def __init__(self, upsample_mode:str="bilinear"):
		"""
		A Full DPNUnet architecture module that takes in a 3x512x512 tensor,
		and outputs a 1x512x512 tensor.

		Arguments it takes are:
		1. upsample_mode (str, optional): Upsampling mode. Default is 'nearest'
		"""
		super(PaperDPNUnet, self).__init__()
		self.input_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
		self.pooling = nn.MaxPool2d(3, stride=2, padding=1)

		self.down_layer1 = DownLayer(64, 288, 96, 96, 272, 256, 3)
		self.down_layer2 = DownLayer(336, 576, 192, 192, 544, 512, 4, True)
		self.down_layer3 = DownLayer(704, 1072, 384, 384, 1048, 1024, 20)
		self.down_layer4 = DownLayer(1552, 2304, 768, 768, 2176, 2048, 3, True)
		
		self.center_block = ConvBNReLU(2688, 2688)

		self.up_layer4 = UpLayer(4240, 704, upsample_mode=upsample_mode)
		self.up_layer3 = UpLayer(1408, 336, upsample_mode=upsample_mode)
		self.up_layer2 = UpLayer(672, 64, upsample_mode=upsample_mode)
		self.up_layer1 = UpLayer(128, 64, upsample_mode=upsample_mode)
		
		self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)
		self.final_conv_block = ConvBNReLU(64, 1)
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
		down_layer1 = self.pooling(x1)
		down_layer1 = self.down_layer1(x1)
		# down_layer2 = self.down_layer2(down_layer1)
		down_layer2 = checkpoint.checkpoint(self.custom_checkpoint_call(self.down_layer2), down_layer1, use_reentrant=False)
		down_layer3 = self.down_layer3(down_layer2)
		# down_layer4 = self.down_layer4(down_layer3)
		down_layer4 = checkpoint.checkpoint(self.custom_checkpoint_call(self.down_layer4), down_layer3, use_reentrant=False)
		
		# Center block
		down_layer4 = self.center_block(down_layer4)
		
		# Up layers (with skip connections)
		# up_layer4 = self.up_layer4(down_layer4, down_layer3)
		x = checkpoint.checkpoint(self.custom_checkpoint_call(self.up_layer4, doubleInput=True), down_layer4, down_layer3, use_reentrant=False)
		x = self.up_layer3(x, down_layer2)
		# up_layer2 = self.up_layer2(up_layer3, down_layer1)
		x = checkpoint.checkpoint(self.custom_checkpoint_call(self.up_layer2, doubleInput=True), x, down_layer1, use_reentrant=False)
		x = self.up_layer1(x, x1)
		# Output
		x = self.upsample(x)
		x = self.final_conv_block(x)
		x = self.sigmoid(x)
		return x