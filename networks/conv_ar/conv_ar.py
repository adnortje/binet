# imports
import torch
import torch.nn as nn
from numbers import Number
import networks.functional as f
from layers import ConvBinarizer
from networks.conv_ar.conv_encoder import ConvEncoder
from networks.conv_ar.conv_decoder import ConvDecoder


"""
Class Convolutional Autoencoder (Additive Reconstruction):

    Method:
        create(self, itrs, patch, bn):
            used to create a Convolutional Residual Autoencoder
            Args:
                itrs (int): number autoencoder iterations
                p_s  (int): input patch size that model is trained on
                b_n  (int): bits in bottle-neck layer
            Return:
                Convolutional Residual Autoencoder (nn.Module)
"""


class ConvAR(nn.Module):

    def __init__(self, itrs, p_s, b_n):

        super(ConvAR, self).__init__()

        # model name
        self.display_name = "ConvAR"
        self.name = "ConvAR"

        # def p_s
        if isinstance(p_s, Number):
            self.p_s = (int(p_s), int(p_s))
        else:
            self.p_s = p_s
        self.p_h, self.p_w = self.p_s
        assert (self.p_w % 2 == self.p_h % 2 == 0), "p_s % 2 != 0"

        # calc bnD
        self.b_n = int(b_n)
        self.bnd = self._calc_bnd()

        if self.bnd == 0:
            raise ValueError("Too Few Bits")

        # Cell of Conv AR system
        class _AutoencoderCell(nn.Module):

            def __init__(self, bnd):
                super(_AutoencoderCell, self).__init__()

                self.bnd = bnd

                # def: encoder & decoder network
                self.enc = ConvEncoder()
                self.bin = ConvBinarizer(
                    in_dim=512,
                    bnd=self.bnd
                )
                self.dec = ConvDecoder(
                    bnd=self.bnd
                )

            def forward(self, x):
                # encode and decode
                x = self.dec(self.bin(self.enc(x)))
                return x

        # ModuleList of Cells
        self.itrs = itrs
        self.ae_sys = nn.ModuleList(
            [
                _AutoencoderCell(self.bnd) for _ in range(self.itrs)
            ]
        )

    def forward(self, r):

        losses = []

        for i in range(self.itrs):
            # encode & decode
            dec, _ = self.ae_sys[i](r)

            # calculate L1 loss
            r = r - dec
            losses.append(r.abs().mean())

            # detach r after after each iteration
            r = r.detach()

        # sum residual loss
        loss = sum(losses) / self.itrs

        return loss

    def encode_decode(self, r, itrs):

        # encodes batch of images r (B, C, h, w)

        if self.itrs < itrs:
            raise ValueError("itrs > Training Iterations")

        with torch.no_grad():
            # extract original image dimensions
            img_size = r.size()[2:]

            # convert images to patches
            r = f.sliding_window(
                input=r,
                kernel_size=self.p_s,
                stride=self.p_s
            )

            # init decoded patches to zero
            patches = 0.0

            for i in range(itrs):
                # encode & decode patches
                dec = self.ae_sys[i](r)
                r = r - dec
                # sum across residuals
                patches += dec

            # clamp patch values [-1,1]
            patches = patches.clamp(-1, 1)

            # reshape patches to images
            images = f.refactor_windows(
                windows=patches,
                output_size=img_size
            )

        return images

    def _calc_bnd(self):
        s = (self.p_w // 16) * (self.p_h // 16)
        return self.b_n // s