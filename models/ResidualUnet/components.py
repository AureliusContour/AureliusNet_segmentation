import torch
from torch import nn

class IdentityResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels=None) -> None:
		super().__init__()
		if not  mid_channels:
			mid_channels = out_channels
		
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(out_channels)
		)
		
		self.relu = nn.ReLU(inplace=True)
	
	def forward(self, x):
		x1 = self.conv1(x)
		x1 = self.conv2(x1)
		x1 = torch.add(x1, x)
		x1 = self.relu(x1)
		return x1
	
class IdentityResidualBlock_V2(nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels=None) -> None:
		super().__init__()
		if not  mid_channels:
			mid_channels = out_channels
		
		self.conv1 = nn.Sequential(
			nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1)
		)

		self.conv2 = nn.Sequential(
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1)
		)
	
	def forward(self, x):
		x1 = self.conv1(x)
		x1 = self.conv2(x1)
		x1 = torch.add(x1, x)
		return x1

class ProjectionResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels=None, stride=2) -> None:
		super().__init__()
		if not  mid_channels:
			mid_channels = out_channels
		
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=stride),
			nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1),
			nn.BatchNorm2d(out_channels)
		)

		self.conv_shortcut = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride),
			nn.BatchNorm2d(out_channels)
		)
		
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		shortcut = self.conv_shortcut(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = torch.add(x, shortcut)
		x = self.relu(x)
		return x
	
class ProjectionResidualBlock_V2(nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels=None, stride=2) -> None:
		super().__init__()
		if not  mid_channels:
			mid_channels = out_channels
		
		self.conv1 = nn.Sequential(
			nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),			
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=stride)
		)

		self.conv2 = nn.Sequential(
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),	
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1)
		)

		self.conv_shortcut = nn.Sequential(
			nn.BatchNorm2d(in_channels),
			nn.ReLU(inplace=True),	
			nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)
		)
		

	def forward(self, x):
		shortcut = self.conv_shortcut(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = torch.add(x, shortcut)
		return x
	
class ResidualDownLayer(nn.Module):
	def __init__(self, in_channels, out_channels, reps) -> None:
		super().__init__()

		self.proj = ProjectionResidualBlock(in_channels, out_channels, stride=2)

		self.id = nn.ModuleList([
			IdentityResidualBlock(in_channels=out_channels, out_channels=out_channels) for _ in range(reps - 1)
		])
	
	def forward(self, x):
		x = self.proj(x)
		for id_block in self.id:
			x = id_block(x)
		return x

class ResidualDownLayer_V2(nn.Module):
	def __init__(self, in_channels, out_channels, reps) -> None:
		super().__init__()

		self.proj = ProjectionResidualBlock_V2(in_channels, out_channels, stride=2)

		self.id = nn.ModuleList([
			IdentityResidualBlock_V2(in_channels=out_channels, out_channels=out_channels) for _ in range(reps - 1)
		])
	
	def forward(self, x):
		x = self.proj(x)
		for id_block in self.id:
			x = id_block(x)
		return x

class ResidualUpLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.proj = ProjectionResidualBlock(in_channels, out_channels, stride=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.proj(x)
	
class ResidualUpLayer_V2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.proj = ProjectionResidualBlock_V2(in_channels, out_channels, stride=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.proj(x)
	
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
	
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)