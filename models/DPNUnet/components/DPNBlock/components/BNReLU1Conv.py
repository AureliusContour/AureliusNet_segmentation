# Library dependencies
from torch import nn

# BN + ReLU + 1x1 conv
class BNReLU1Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        A module composing of Batch Normalization + ReLU + 1x1 Conv
        """
        super(BNReLU1Conv, self).__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_ch,out_ch, kernel_size=1, padding=0)
		
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
    
