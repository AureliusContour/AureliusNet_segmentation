from .components import *

class ResidualUnet(nn.Module):
	def __init__(self, n_channels) -> None:
		super(ResidualUnet, self).__init__()

		self.in_conv = DoubleConv(n_channels, 64)
		self.out_conv = OutConv(64, 1)

		self.down1 = ResidualDownLayer(64, 64, 3)
		self.down2 = ResidualDownLayer(64, 128, 4)
		self.down3 = ResidualDownLayer(128, 256, 6)
		self.down4 = ResidualDownLayer(256, 512, 3)
		
		self.up4 = ResidualUpLayer(512 + 256, 256)
		self.up3 = ResidualUpLayer(256 + 128, 128)
		self.up2 = ResidualUpLayer(128 + 64, 64)
		self.up1= ResidualUpLayer(64 + 64, 64)

	def forward(self, x):
		x0 = self.in_conv(x)
		x1 = self.down1(x0)
		x2 = self.down2(x1)
		x3 = self.down3(x2)
		x4 = self.down4(x3)
		x = self.up4(x4, x3)
		x = self.up3(x, x2)
		x = self.up2(x, x1)
		x = self.up1(x, x0)
		x = self.out_conv(x)
		x = torch.sigmoid(x)
		return x