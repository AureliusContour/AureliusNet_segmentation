import torch.nn as nn


class BNReLU3ConvDropout(nn.Module):
    """
    (Batch Normalization) => (reLU) => (3x3 Conv2D) => (Dropout)

    Args:
        in_channel (int): Number of input channels
        out_channel (int): Number of output channels
        kernel_size (int, optional): Size of the convolutional kernel. Default is 3
        padding (int, optional): Amount of zero-padding added to both sides of the input. Default is 0
        stride (int, optional): Stride of the convolution. Default is 1
        dropout (float, optional): Dropout probability. Default is 0.5
    """

    def __init__(self, in_channel: int, out_channel: int, kernel_size: int = 3, padding: int = 1, stride: int = 1, dropout: float = 0.5) -> None:
        super().__init__()

        self.batch_norm = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x
