import torch
import torch.nn as nn

class AttentionGate(nn.Module):
	def __init__(self, in_channels, gate_channels):
		super().__init__()
		self.conv_x = nn.Conv2d(in_channels, gate_channels, kernel_size=1, stride=2, bias=False)
		self.conv_g = nn.Conv2d(gate_channels, gate_channels, kernel_size=1)
		self.conv_to_one = nn.Conv2d(gate_channels, 1, kernel_size=1)
		self.relu = nn.ReLU(inplace=True)
		self.upsample = nn.Upsample(scale_factor=2)

	def forward(self, x, gate_signal):
		theta_x = self.conv_x(x)
		gate_signal = self.conv_g(gate_signal)
		gate_signal = torch.add(theta_x, gate_signal)
		gate_signal = self.relu(gate_signal)
		gate_signal = self.conv_to_one(gate_signal)
		gate_signal = torch.sigmoid(gate_signal)
		gate_signal = self.upsample(gate_signal)
		return torch.mul(x, gate_signal)
    
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
    
class DownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class UpLayer(nn.Module):
	def __init__(self, skip_channels, in_channels, out_channels):
		super().__init__()
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.conv = DoubleConv(skip_channels + in_channels, out_channels, (skip_channels + in_channels) // 2)
		self.attention_gate = AttentionGate(skip_channels, in_channels)

	def forward(self, x1, x2):
		x2 = self.attention_gate(x2, gate_signal=x1)
		x1 = self.up(x1)
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)