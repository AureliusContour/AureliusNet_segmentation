import torch.nn.functional as F
import torch

from .components import *


class InceptionUnet(nn.Module):
    def __init__(self, n_channels):
        super(InceptionUnet, self).__init__()
        self.n_channels = n_channels

        self.block1 = InceptionDoubleConv(64, 32)
        self.block2 = InceptionDoubleConv(128, 64)
        self.block3 = InceptionDoubleConv(256, 128)
        self.block4 = InceptionDoubleConv(512, 128)

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownLayer(64, 128)
        self.down2 = DownLayer(128, 256)
        self.down3 = DownLayer(256, 512)
        self.down4 = DownLayer(512, 512)

        self.up1 = UpInceptionLayer(1024+512, 128)
        self.up2 = UpInceptionLayer(896, 64)
        self.up3 = UpInceptionLayer(448, 16)
        self.up4 = UpInceptionLayer(208, 16)
        self.outc = OutConv(16, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        block1 = self.block1(x1)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        x = self.up1(x5, x4, block4)
        x = self.up2(x, x3, block3)
        x = self.up3(x, x2, block2)
        x = self.up4(x, x1, block1)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x