import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from .DecoderBlock import DecoderBlock
from .DoubleConv import DoubleConv

class UpLayer(nn.Module):
	def __init__(self, in_channels, out_channels, upsample_mode="bilinear", checkpoint=False):
		super().__init__()
		self.up = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True)
		self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
		self.checkpoint = checkpoint

	def custom_checkpoint_call(self, module):
		def custom_forward(*inputs):
			inputs = module(inputs[0])
			return inputs
		return custom_forward

	def forward(self, x1, x2):
		x1 = self.up(x1)
		x = torch.cat([x2, x1], dim=1)
		if self.checkpoint:
			x =	checkpoint.checkpoint(self.custom_checkpoint_call(self.conv), x, use_reentrant=False)
		else:
			x = self.conv(x)
		return x
	
class AureliusUpLayer(nn.Module):
	def __init__(self, in_channels, out_channels, upsample_mode="bilinear", checkpoint=False):
		super().__init__()
		self.up = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True)
		self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
		self.checkpoint = checkpoint

	def custom_checkpoint_call(self, module):
		def custom_forward(*inputs):
			inputs = module(inputs[0])
			return inputs
		return custom_forward

	def forward(self, x1, x2):
		x1 = self.up(x1)

		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		if self.checkpoint:
			x =	checkpoint.checkpoint(self.custom_checkpoint_call(self.conv), x, use_reentrant=False)
		else:
			x = self.conv(x)
			
		return x