# Library dependencies
import torch
from torch import nn
from .components import BNReLU1Conv
from .components import BNReLU3Conv

# DPN Block
class DPNBlock(nn.Module):
	def __init__(self, in_ch:int, size_conv_a:int, size_conv_b:int, size_conv_c:int, split_size:int) -> None:
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
		self.split_size = split_size
		self.init_split_size_small = in_ch - split_size
		self.end_split_size_small = size_conv_c - split_size

	def forward(self, x):
		split1_top, split1_bottom = torch.split(x, [self.split_size, self.init_split_size_small], dim=1)
		x = self.brc1a(x)
		x = self.brc3b(x)
		x = self.brc1c(x)
		split2_top, split2_bottom = torch.split(x, [self.split_size, self.end_split_size_small], dim=1)
		summed_top = torch.add(split2_top, split1_top)
		concat_bottom = torch.concat((split2_bottom, split1_bottom), dim=1)
		x = torch.concat((summed_top, concat_bottom), dim=1)
		return x

class InitDPNBlock(nn.Module):
	def __init__(self, in_ch:int, 
				out_conv1_1_ch:int, 
				out_conv1_2_ch:int, 
				out_conv3_ch:int, 
				out_conv1_3_ch:int, 
				split_size:int) -> None:
		"""
		A DPN Block that takes in the
		1. input channel, 
		2. output channel of the alternate route 1x1 conv layer,
		3. output channel of first 1x1 conv layer,
		4. output channel of the following 3x3 conv layer,
		5. output channel of the last 1x1 conv layer,
		6. and the split size for the initial split (bottom/smaller portion).
		"""
		super(InitDPNBlock, self).__init__()
		self.BN_ReLU_Conv1_1 = BNReLU1Conv(in_ch, out_conv1_1_ch, stride=2)
		self.BN_ReLU_Conv1_2 = BNReLU1Conv(in_ch, out_conv1_2_ch)
		self.BN_ReLU_Conv3 = BNReLU3Conv(out_conv1_2_ch, out_conv3_ch, stride=2)
		self.BN_ReLU_Conv1_3 = BNReLU1Conv(out_conv3_ch, out_conv1_3_ch)
		self.split_size = split_size
		self.split_size_small = out_conv1_3_ch - split_size
		self.split_alternate_size_small = out_conv1_1_ch - split_size

	def forward(self, x):
		x_alternate = self.BN_ReLU_Conv1_1(x)
		x = self.BN_ReLU_Conv1_2(x)
		x = self.BN_ReLU_Conv3(x)
		x = self.BN_ReLU_Conv1_3(x)
		split_big, split_small = torch.split(x, [self.split_size, self.split_size_small], dim=1)
		split_alt_big, split_alt_small = torch.split(x_alternate, [self.split_size, self.split_alternate_size_small], dim=1)
		summed_big = torch.add(split_big, split_alt_big)
		concat_small = torch.concat((split_small, split_alt_small), dim=1)
		x = torch.concat((summed_big, concat_small), dim=1)
		return x