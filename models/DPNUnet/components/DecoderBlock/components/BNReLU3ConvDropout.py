import torch.nn as nn

class BNReLU3ConvDropout(nn.Module):
    """(Batch Normalization) => (reLU) => (3x3 Conv2D) => (Dropout)"""
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1, dropout=0.5) -> None:
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