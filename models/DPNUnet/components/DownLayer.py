# Library dependencies
from torch import nn
from .DPNBlock import DPNBlock
import torch.utils.checkpoint as checkpoint

# Down Layer (DPN Block)
class DownLayer(nn.Module):
	def __init__(self, in_ch, size_conv_a, size_conv_b, size_conv_c, split_size, num_of_blocks, checkpoint=False):
		"""
		A Down Layer that composes of an initial 3x3 Max Pooling (stride 2),
		followed by a number of DPN Blocks that takes in the arguments:
		1. input channel, 
		2. output channel of first 1x1 conv layer,
		3. output channel of the following 3x3 conv layer,
		4. output channel of the last 1x1 conv layer,
		5. and the split size for the initial split (bottom/smaller portion),
		6. num of DPN Blocks.
		"""
		super(DownLayer, self).__init__()
		self.num_of_blocks = num_of_blocks
		self.max_pooling = nn.MaxPool2d(3, stride=2, padding=1)
		self.initDPN = DPNBlock(in_ch, size_conv_a, size_conv_b, size_conv_c, split_size)
		self.nextDPN = nn.ModuleList([
			DPNBlock(size_conv_c + split_size, size_conv_a, size_conv_b, size_conv_c, split_size) for _ in range(num_of_blocks - 1)
		])
		self.checkpoint = checkpoint

	def custom_checkpoint_call(self, module):
		def custom_forward(*inputs):
			inputs = module(inputs[0])
			return inputs
		return custom_forward
	
	def forward(self, x):
		x = self.max_pooling(x)
		x = self.initDPN(x)
		for i in range(self.num_of_blocks - 1):
			x = self.nextDPN[i](x)
			# if self.checkpoint and ( (i + 2) % round(self.num_of_blocks ** 0.5) ) == 0:
			# 	x = checkpoint.checkpoint(self.custom_checkpoint_call(self.nextDPN[i]), x, use_reentrant=False)
			# else:
			# 	x = self.nextDPN[i](x)
		return x
