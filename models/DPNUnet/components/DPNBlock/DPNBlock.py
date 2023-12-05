# Library dependencies
import torch
from torch import nn
from .components.BNReLU1Conv import BNReLU1Conv
from .components.BNReLU3Conv import BNReLU3Conv

# DPN Block
class DPNBlock(nn.Module):
	def __init__(self, in_ch, size_conv_a, size_conv_b, size_conv_c, split_size):
		"""
		A DPN Block that takes in the
		1. input channel, 
		2. output channel of first 1x1 conv layer,
		3. output channel of the following 3x3 conv layer,
		4. output channel of the last 1x1 conv layer,
		5. and the split size for the initial split (bottom/smaller portion).
		"""
		super(DPNBlock, self).__init__()
		self.brc1a = BNReLU1Conv(in_ch, size_conv_a)
		self.brc3b = BNReLU3Conv(size_conv_a, size_conv_b)
		self.brc1c = BNReLU1Conv(size_conv_b, size_conv_c)
		self.split1_size_bottom = split_size
		self.split1_size_top = in_ch - split_size
		self.split2_size_top = self.split1_size_top
		self.split2_size_bottom = size_conv_c - self.split1_size_top

	def forward(self, x):
		conv1 = self.brc1a(x)
		conv2 = self.brc3b(conv1)
		conv3 = self.brc1c(conv2)
		split1_top, split1_bottom = torch.split(x, [self.split1_size_top, self.split1_size_bottom], dim=1)
		split2_top, split2_bottom = torch.split(conv3, [self.split2_size_top, self.split2_size_bottom], dim=1)
		summed_top = torch.add(split2_top, split1_top)
		concat_bottom = torch.concat((split2_bottom, split1_bottom), dim=1)
		concat_all = torch.concat((summed_top, concat_bottom), dim=1)
		return concat_all

