import torch
import torch.nn as nn
from components.BNReLU3ConvDropout import BNReLU3ConvDropout


class DecoderBlock(nn.Module):
    """
    A Decoder block used in a DPN U Net architecture

    Args:
        in_channel (int): Number of input channels
        out_channel (int): Number of output channels

    Attributes:
        bn_relu_conv_dropout_front (BNReLU3ConvDropout): First BNReLU3ConvDropout layer
        bn_relu_conv_dropout_mid (BNReLU3ConvDropout): Intermediary BNReLU3ConvDropout layer
        BNReLU3ConvDropout_end (BNReLU3ConvDropout): Final BNReLU3ConvDropout layer
    """

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()

        self.bn_relu_conv_dropout_front = BNReLU3ConvDropout(
            in_channel=in_channel, out_channel=out_channel)

        self.bn_relu_conv_dropout_mid = BNReLU3ConvDropout(
            in_channel=out_channel, out_channel=out_channel)

        self.bn_relu_conv_dropout_end = BNReLU3ConvDropout(
            in_channel=int(out_channel*1.5), out_channel=out_channel)

    def forward(self, x):
        # First Split
        x = self.bn_relu_conv_dropout_front(x)
        split_1_first_part, split_1_second_part = torch.tensor_split(
            x, 2, dim=1)

        # Second Split
        x = self.bn_relu_conv_dropout_mid(x)
        x = self.bn_relu_conv_dropout_mid(x)
        split_2_first_part, split_2_second_part = torch.tensor_split(
            x, 2, dim=1)

        # Point to Point sum
        ptop_sum = torch.add(split_1_second_part, split_2_second_part)

        x = torch.cat((split_1_first_part, split_2_first_part, ptop_sum), dim=1)
        x = self.bn_relu_conv_dropout_end(x)

        return x
