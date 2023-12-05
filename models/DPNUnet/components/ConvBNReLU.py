# Library dependencies
from torch import nn

# 3x3 conv + BN + ReLU
class ConvBNReLU(nn.Module):
	def __init__(self, in_ch, out_ch):
		"""
		A module composing of 3x3 Conv + Batch Normalization + ReLU
		"""
		super(ConvBNReLU, self).__init__()
		self.conv = nn.Conv2d(in_ch,out_ch, kernel_size=3, padding=1)
		self.bn = nn.BatchNorm2d(out_ch)
		self.relu = nn.ReLU()
		
	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x

