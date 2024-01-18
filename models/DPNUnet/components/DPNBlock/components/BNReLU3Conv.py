# Library dependencies
from torch import nn

# BN + ReLU + 3x3 conv
class BNReLU3Conv(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, stride=1):
        """
        A module composing of Batch Normalization + ReLU + 1x1 Conv
        """
        super(BNReLU3Conv, self).__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch,out_ch, kernel_size=3, stride=stride, padding=1)
		
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
    
