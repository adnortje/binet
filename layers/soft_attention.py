# imports
import torch
import torch.nn as nn
from .conv_bin import ConvBinarizer

"""
Soft Attention Network
        
    Args:
        in_dim (int) : input channel dimension
"""


class SoftAttention(nn.Module):

    def __init__(self, in_dim, attn_bnd):
        super(SoftAttention, self).__init__()

        self.attn = nn.Sequential(

            nn.Conv2d(
                in_channels=in_dim,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),

            nn.ReLU(),

            ConvBinarizer(
                in_dim=128,
                bnd=attn_bnd
            ),

            nn.Conv2d(
                in_channels=attn_bnd,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),

            nn.Conv2d(
                in_channels=128,
                out_channels=in_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )

        )

    def forward(self, x, feat):

        # extract x size
        b, c, h, w = x.size()

        # calc attention weights
        attn_weights = self.attn(feat)
        attn_weights = nn.Softmax(dim=1)(
            attn_weights.view(b, -1)
        )
        attn_weights = attn_weights.view(b, c, h, w)

        # apply attention mask
        x_attn = x * attn_weights

        # concat attention weights
        x = torch.cat([x, x_attn], dim=1)

        return x