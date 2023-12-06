import torch
import torch.nn as nn
from .DecoderBlock import DecoderBlock


class UpLayer(nn.Module):
	def __init__(self, in_channel: int, skip_channel: int, out_channel: int, upsample_mode: str = 'nearest') -> None:
		"""
		UpLayer that combines features from previous UpLayer and skip connection
		into the decoder block

		Args:
				in_channel (int): Number of input channels from previous layer
				skip_channel (int): Number of input channels from skip connection
				out_channel (int): Number of output channels
				upsample_mode (str, optional): Upsampling mode. Default is 'nearest'
		"""
		super().__init__()

		self.decoder = DecoderBlock(
			in_channel=in_channel+skip_channel, out_channel=out_channel)
		self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)

	def forward(self, x1, x2):
		"""
		Receives 2 input from previous UpLayer and the skip connection

		Args:
			x1 (torch.Tensor): input tensor from previous UpLayer
			x2 (torch.Tensor): input tensor from the skip connection
		"""
		x1 = self.upsample(x1)
		x1 = torch.cat((x1, x2), dim=1)
		x1 = self.decoder(x1)

		return x1