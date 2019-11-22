# imports
import torch
import warnings
import torch.nn as nn
import networks.functional as f
import torch.nn.functional as F
from layers import ConvBinarizer
from torch.nn.modules.utils import _pair
from networks.conv_ar import ConvAR, ConvEncoder, ConvDecoder


"""
BINetAR

    Convolutional BINet using Additive Reconstruction Framework.

        Args: 
            itrs (int) : encoding iterations
            p_s  (int) : patch size
            b_n  (int) : bits in bottle-neck
            n_p  (int) : number of patches that make up a square grid


"""


class BINetAR(nn.Module):

    def __init__(self, itrs, p_s, b_n):
        super(BINetAR, self).__init__()

        # def model name
        self.name = "BINetAR"
        self.display_name = "BINetAR"

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

        # bits in bottle-neck
        self.b_n = b_n

        # number of encoding iterations
        self.itrs = itrs

        # inpaint encoder
        self.inp_encoder = ConvEncoder()

        # inpaint binarizer
        self.inp_binarizer = ConvBinarizer(
            in_dim=512,
            bnd=self._calc_bnd()
        )

        # inpaint decoder
        self.inp_decoder = ConvDecoder(
            bnd=9*self._calc_bnd()
        )

        # residual autoencoder
        self.res_auto = ConvAR(
            itrs=self.itrs-1,
            p_s=self.p_s,
            b_n=self.b_n
        )

    def _inpaint_encode(self, x, is_forward=False):

        # extract init dimensions
        x_h, x_w = x.size()[2:]

        # images -> patches
        x = f.sliding_window(
            input=x,
            kernel_size=self.p_s,
            stride=self.p_s
        )

        # encode
        x = self.inp_encoder(x)

        # binarize
        x = self.inp_binarizer(x)

        if is_forward:
            # training
            x = x.view(-1, self.inp_decoder.bnd, *x.size()[2:])

        else:
            # inference
            x = f.refactor_windows(
                windows=x,
                output_size=(x_h//16, x_w//16)
            )

        return x

    def _inpaint_decode(self, x):

        # init size
        x_h, x_w = x.size()[2:]
        init_size = (16*x_h, 16*x_w)

        # replication pad
        x = F.pad(
            input=x,
            pad=(
                self.p_w // 16,
                self.p_w // 16,
                self.p_h // 16,
                self.p_h // 16
            ),
            mode='replicate'
        )

        # bit image -> bit grids
        x = f.sliding_window(
            input=x,
            kernel_size=(
                self.g_h // 16,
                self.g_w // 16
            ),
            stride=(
                self.p_h // 16,
                self.p_w // 16
            )
        )

        # bit grids -> bit patches
        x = f.sliding_window(
            input=x,
            kernel_size=(
                self.p_h // 16,
                self.p_w // 16
            ),
            stride=(
                self.p_h // 16,
                self.p_w // 16
            )
        )

        # concat bits
        x = x.view(-1, self.inp_decoder.bnd, *x.size()[2:])

        # decode patches
        x = self.inp_decoder(x)

        # patches -> image
        x = f.refactor_windows(
            windows=x,
            output_size=init_size
        )

        return x

    def forward(self, r):

        losses = []

        # inpaint encode
        enc = self._inpaint_encode(r, is_forward=True)

        # decode central patches
        dec = self.inp_decoder(enc)

        center_patches = r[:, :, self.p_h:2*self.p_h, self.p_w:2*self.p_w]

        # L1 Loss
        losses.append(
            nn.L1Loss()(dec, target=center_patches)
        )

        r = center_patches - dec

        for i in range(self.itrs-1):

            # detach r
            r = r.detach()

            # encode & decode residual error
            dec = self.res_auto.ae_sys[i](r)

            # append L1 loss
            losses.append(
                nn.L1Loss()(dec, target=r)
            )

            # residual error
            r = r - dec

        # sum & normalize residual loss
        loss = sum(losses) / self.itrs

        return loss

    def encode_decode(self, r, itrs):

        if self.itrs < itrs:
            raise ValueError("Specified itrs > Model itrs")

        # inpaint encode & decode
        enc = self._inpaint_encode(r)
        dec = self._inpaint_decode(enc)

        # inpaint residual
        r = r - dec

        # encode & decode residual error
        if itrs > 1:
            dec += self.res_auto.encode_decode(r, itrs-1)

        # clamp output [-1, 1]
        dec = dec.clamp(-1, 1)

        return dec

    def _calc_bnd(self):
        # inpainting network bottleneck depth
        s = (self.p_w // 16)*(self.p_h // 16)
        return self.b_n//s
