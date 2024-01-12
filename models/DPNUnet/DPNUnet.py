# Library dependencies
from torch import nn
import torch.utils.checkpoint as checkpoint
from .components import UpLayer
from .components import DownLayer
from .components import ConvBNReLU

# DPN-Unet architecture
class DPNUnet(nn.Module):
	def __init__(self, upsample_mode:str="nearest", is_small:bool=True, 
			  dpn_block_count_1:int=3, 
			  dpn_block_count_2:int=4, 
			  dpn_block_count_3:int=8, 
			  dpn_block_count_4:int=3):
		"""
		A Full DPNUnet architecture module that takes in a 3x512x512 tensor,
		and outputs a 1x512x512 tensor.

		Arguments it takes are:
		1. upsample_mode (str, optional): Upsampling mode. Default is 'nearest'
		"""
		div_fctr = 2 if is_small else 1
		super(DPNUnet, self).__init__()
		self.input_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
		self.down_layer1 = DownLayer(64, 48, 48, 128, 16, dpn_block_count_1)
		self.up_layer1 = UpLayer(134, 64, 64, upsample_mode=upsample_mode)
		self.down_layer2 = DownLayer(134, 96, 96, 256, 32, dpn_block_count_2, True)
		self.up_layer2 = UpLayer(288, 134, 134, upsample_mode=upsample_mode)
		self.down_layer3 = DownLayer(288, 192, 192, 512, 64, dpn_block_count_3)
		self.up_layer3 = UpLayer(576, 288, 288, upsample_mode=upsample_mode)
		self.down_layer4 = DownLayer(576, 384, 384, 1024, 128, dpn_block_count_4, True)
		self.up_layer4 = UpLayer(1152, 576, 576, upsample_mode=upsample_mode)
		self.center_block = ConvBNReLU(1152, 1152)
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
		x = self.input_conv(x)
		# Down layers
		down_layer1 = self.down_layer1(x)
		# down_layer2 = self.down_layer2(down_layer1)
		down_layer2 = checkpoint.checkpoint(self.custom_checkpoint_call(self.down_layer2), down_layer1, use_reentrant=False)
		down_layer3 = self.down_layer3(down_layer2)
		# down_layer4 = self.down_layer4(down_layer3)
		down_layer4 = checkpoint.checkpoint(self.custom_checkpoint_call(self.down_layer4), down_layer3, use_reentrant=False)
		# Center block
		down_layer4 = self.center_block(down_layer4)
		# Up layers (with skip connections)
		# up_layer4 = self.up_layer4(down_layer4, down_layer3)
		up_layer4 = checkpoint.checkpoint(self.custom_checkpoint_call(self.up_layer4, doubleInput=True), down_layer4, down_layer3, use_reentrant=False)
		up_layer3 = self.up_layer3(up_layer4, down_layer2)
		up_layer2 = self.up_layer2(up_layer3, down_layer1)
		up_layer2 = checkpoint.checkpoint(self.custom_checkpoint_call(self.up_layer2, doubleInput=True), up_layer3, down_layer1, use_reentrant=False)
		up_layer1 = self.up_layer1(up_layer2, x)
		# Output
		x = self.upsample(up_layer1)
		x = self.final_conv_block(x)
		x = self.sigmoid(x)
		return x

