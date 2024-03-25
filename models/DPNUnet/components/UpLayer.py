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
		# self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
		self.decode =DecoderBlock(in_channels, out_channels)
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
			x =	checkpoint.checkpoint(self.custom_checkpoint_call(self.decode), x, use_reentrant=False)
		else:
			x = self.decode(x)
		return x