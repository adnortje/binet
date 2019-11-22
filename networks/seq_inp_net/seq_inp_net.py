# imports
import os
import torch
import warnings
import torch.nn as nn
from torch.nn.modules.utils import _pair
from networks.conv_ar import ConvAR


"""
Sequential Inpainting Network

    used to compare binary inpainting to sequential inpainting from decompressed patches
    
    Args:
        p_s (int) : patch size
        b_n (int) : bottleneck-depth   
         
"""


class SINet(nn.Module):

    def __init__(self, p_s, b_n, auto_weights):
        super(SINet, self).__init__()

        # def model name
        self.name = "SINet"
        self.display_name = "SINet"

        # def patch size
        self.p_s = _pair(p_s)
        self.p_h, self.p_w = self.p_s
        if self.p_h % 2 != 0 or self.p_w % 2 != 0:
            warnings.warn("Patch size is not divisible by 2, patch will be truncated")

        # def grid size
        self.g_s = (
            3*self.p_h,
            3*self.p_w
        )
        self.g_h, self.g_w = self.g_s

        # def bits in bottle-neck
        self.b_n = b_n

        # def conv autoencoder
        self.conv_auto = ConvAR(
            itrs=16,
            p_s=self.p_s,
            b_n=self.b_n
        )

        # check weight file exists
        auto_weights = os.path.expanduser(auto_weights)

        if not os.path.isfile(auto_weights):
            raise FileNotFoundError("Pre-trained weights file D.N.E!")

        # load pre-trained weights
        self.conv_auto.load_state_dict(
            torch.load(auto_weights)
        )

        # freeze compression weights
        for param in self.conv_auto.parameters():
            param.requires_grad = False

        # inpainting channels = no. patches  * RGB
        self.inpaint_channels = 3 * 6
        # def inpainting network
        self.inp_net = nn.Sequential(

            nn.Conv2d(
                in_channels=self.inpaint_channels,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=3,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                bias=True
            ),

            nn.Tanh()
        )

    def forward(self, x):

        # discard bottom-row
        x = x[:, :, :-self.p_h, :]
        # center patch
        center_patch = x[:, :, self.p_h:, self.p_w: -self.p_w]

        # encode & decode patches
        x = self.conv_auto.encode_decode(x, itrs=1)

        # partial context mask
        seq_mask = torch.ones_like(x)
        seq_mask[:, :, self.p_h:, self.p_w:] = 0.0
        x = x*seq_mask

        # patches along channel dimension
        x = x.view(-1, self.inpaint_channels, self.p_h,  self.p_w)

        # inpaint center patch
        x = self.inp_net(x)

        # loss
        loss = nn.L1Loss()(x, target=center_patch)
        return loss

    def encode_decode(self, x):

        # discard bottom-row
        x = x[:, :, :-self.p_h, :]

        # encode & decode patches
        x = self.conv_auto.encode_decode(x, itrs=1)

        # partial context mask
        seq_mask = torch.ones_like(x)
        seq_mask[:, :, self.p_h:, self.p_w:] = 0.0
        x = x*seq_mask

        # patches along channel dimension
        x = x.view(-1, self.inpaint_channels, self.p_h,  self.p_w)

        # inpaint center patch
        x = self.inp_net(x)

        return x
