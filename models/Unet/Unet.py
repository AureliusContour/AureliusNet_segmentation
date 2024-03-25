from .components import *

class Unet(nn.Module):
    def __init__(self, n_channels):
        super(Unet, self).__init__()
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (DownLayer(64, 128))
        self.down2 = (DownLayer(128, 256))
        self.down3 = (DownLayer(256, 512))
        self.down4 = (DownLayer(512, 1024 // 2))
        self.up1 = (UpLayer(1024, 512 // 2))
        self.up2 = (UpLayer(512, 256 // 2))
        self.up3 = (UpLayer(256, 128 // 2))
        self.up4 = (UpLayer(128, 64))
        self.outc = (OutConv(64, 1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x
