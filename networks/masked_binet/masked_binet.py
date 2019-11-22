# imports
import torch
import warnings
import torch.nn as nn
import networks.functional as f
from layers import ConvBinarizer
from torch.nn.modules.utils import _pair
from networks.conv_ar import ConvEncoder, ConvDecoder

"""
MaskedBINet

    Convolutional BINet with masked bit region. 
    Model is used for the experimental evaluation of the plausibility of binary inpainting.

        Args: 
            p_s  (int) : patch size
            b_n  (int) : bits in bottle-neck
"""


class MaskedBINet(nn.Module):

    def __init__(self, p_s, b_n):
        super(MaskedBINet, self).__init__()

        # def model name
        self.name = "MaskedBINet"
        self.display_name = "Masked BINet"

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

        # encoder
        self.encoder = ConvEncoder()

        # binary network
        self.binarizer = ConvBinarizer(
            in_dim=512,
            bnd=self._calc_bnd()
        )

        # decoder
        self.decoder = ConvDecoder(
            bnd=self._calc_bnd()*9
        )

    def encode(self, x):

        # grids -> patches
        x = f.sliding_window(
            input=x,
            kernel_size=self.p_s,
            stride=self.p_s
        )

        # encode & binarize
        x = self.encoder(x)
        x = self.binarizer(x)

        return x

    def decode(self, x):

        # mask center bits
        x[range(4, x.size(0), 9), :, :, :] = 0.0

        # reshape bits
        x = x.view(
            -1,
            self.decoder.bnd,
            self.p_h // 16,
            self.p_w // 16
        )

        # decode center patch
        x = self.decoder(x)

        return x

    def forward(self, x):

        # target center patches
        center_patches = x[
            :, :, self.p_h: -self.p_h, self.p_w: -self.p_w
        ]

        # inpainting prediction
        inp_pred = self.decode(self.encode(x))

        # L1 Loss
        loss = nn.L1Loss()(
            inp_pred,
            target=center_patches
        )

        return loss

    def encode_decode(self, x):
        # encode & decode, inpaint output patch

        with torch.no_grad():
            dec = self.decode(self.encode(x))

        return dec

    def load_model(self, save_loc):
        # load weights
        self.load_state_dict(torch.load(save_loc))
        return

    def _calc_bnd(self):
        bnd = self.b_n // ((self.p_w // 16) * (self.p_h // 16))
        return bnd

