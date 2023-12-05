# Library dependencies
from torch import nn
from .DPNBlock import DPNBlock

# Down Layer (DPN Block)
class DownLayer(nn.Module):
	def __init__(self, in_ch, size_conv_a, size_conv_b, size_conv_c, split_size, num_of_blocks):
		"""
		A Down Layer that composes of a number of DPN Blocks that takes in the
		1. input channel, 
		2. output channel of first 1x1 conv layer,
		3. output channel of the following 3x3 conv layer,
		4. output channel of the last 1x1 conv layer,
		5. and the split size for the initial split (bottom/smaller portion),
		6. num of DPN Blocks.
		"""
		super(DownLayer, self).__init__()
		self.num_of_blocks = num_of_blocks
		self.initDPN = DPNBlock(in_ch, size_conv_a, size_conv_b, size_conv_c, split_size)
		self.nextDPN = DPNBlock(size_conv_c + split_size, size_conv_a, size_conv_b, size_conv_c, split_size)

	def forward(self, x):
		output = self.initDPN(x)
		for _ in range(self.num_of_blocks - 1):
			output = self.nextDPN(output)
		return output

